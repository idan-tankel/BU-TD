import copy
import sys
sys.path.append(r'/')
from tqdm.auto import tqdm
from typing import Dict, Union
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import torch
from avalanche.models.utils import avalanche_forward
from avalanche.training.utils import copy_params_dict, zerolike_params_dict
from avalanche.training.plugins import MASPlugin
from torch.utils.data import Dataset
from supp.utils import preprocess
from avalanche.training.templates.supervised import SupervisedTemplate
# TODO - ALSO HERE GET RID OF THE BUMODEL, INSTEAD USE FEAUTURES.

class MyMASPlugin(MASPlugin):
    """
    Memory Aware Synapses (MAS) plugin.

    Similarly to EWC, the MAS plugin computes the importance of each
    parameter at the end of each experience. The approach computes
    importance via a second pass on the dataset. MAS does not require
    supervision and estimates importance using the gradients of the
    L2 norm of the output. Importance is then used to add a penalty
    term to the loss function.

    Technique introduced in:
    "Memory Aware Synapses: Learning what (not) to forget"
    by Aljundi et. al (2018).

    Implementation based on FACIL, as in:
    https://github.com/mmasana/FACIL/blob/master/src/approach/mas.py
    """

    def __init__( self, parser:argparse, verbose:bool = False ):
        """
        :param lambda_reg: hyperparameter weighting the penalty term
               in the loss.
        :param alpha: hyperparameter used to update the importance
               by also considering the influence in the previous
               experience.
        :param verbose: when True, the computation of the influence
               shows a progress bar using tqdm.
        """

        # Init super class
        super().__init__()

        # Regularization Parameters
        self._lambda = parser.mas_lambda
        self.alpha = parser.mas_alpha
        # Model parameters
        self.params: Union[Dict, None] = None
        self.importance: Union[Dict, None] = None
        self.batch_size = parser.bs
        self.device = parser.device
        self.parser = parser
        # Progress bar
        self.verbose = verbose
        self.pretrained_model = parser.pretrained_model
        if parser.pretrained_model:
            self.model_old = copy.deepcopy(parser.model)
            self.old_data = parser.old_dataset
            print("Computing Importances")
            self.importance = self._get_importance(self.model_old, self.old_data, self.batch_size, parser.device)
            print("Done computing Importances")
            self.params = self.params = dict(copy_params_dict(self.model_old.bumodel))

    def _get_importance(self, model:nn.Module, dataset:Dataset, train_mb_size:int,device:Union['cuda','cpu']):

        # Initialize importance matrix
        importance = dict(zerolike_params_dict(model.bumodel))

        '''
        if not strategy.experience:
            raise ValueError("Current experience is not available")

        if strategy.experience.dataset is None:
            raise ValueError("Current dataset is not available")
        '''
        # Do forward and backward pass to accumulate L2-loss gradients
        model.train()
        dataloader = DataLoader(dataset,  batch_size=train_mb_size,)

        # Progress bar
        if self.verbose:
            print("Computing importance")
            dataloader = tqdm(dataloader)

        for _, batch in enumerate(dataloader):
            # Get batch

            # Move batch to device
            batch = preprocess(batch, device)

            # Forward pass
            model.zero_grad()
            '''
            if len(batch) == 2 or len(batch) == 3:
                x, _, t = batch[0], batch[1], batch[-1]
            else:
                raise ValueError("Batch size is not valid")
            '''
            # Move batch to device
            # Forward pass
            model.zero_grad()
            out = model.forward_and_out_to_struct(batch, 0).classifier
            # Average L2-Norm of the output
            loss = torch.norm(out, p="fro", dim=1).mean()
            loss.backward()

            # Accumulate importance
            for name, param in model.bumodel.named_parameters():
                if param.requires_grad:
                    # In multi-head architectures, the gradient is going
                    # to be None for all the heads different from the
                    # current one.
                    if param.grad is not None:
                        importance[name] += param.grad.abs() * len(batch)

        # Normalize importance
        importance = { name: importance[name] / len(dataloader) for name in importance.keys()  }

        return importance

    def before_backward(self, strategy, **kwargs):
        # Check if the task is not the first
        exp_counter = strategy.EpochClock.train_exp_counter
        if exp_counter == 0:
            return

        loss_reg = 0.0

        # Check if properties have been initialized
        if not self.importance:
            raise ValueError("Importance is not available")
        if not self.params:
            raise ValueError("Parameters are not available")
        if not strategy.loss:
            raise ValueError("Loss is not available")

        # Apply penalty term
        for name, param in strategy.model.bumodel.named_parameters():

            if name in self.importance.keys():
                loss_reg += torch.sum(
                    self.importance[name] * (param - self.params[name]).pow(2)
                )

        # Update loss
     #   print(self._lambda * loss_reg)
        strategy.loss += self._lambda * loss_reg

    def before_training(self, strategy:SupervisedTemplate, **kwargs):
        # Parameters before the first task starts

        if strategy.EpochClock.just_initialized and not self.pretrained_model:

            if not self.params:
                self.params = dict(copy_params_dict(strategy.model))

            # Initialize Fisher information weight importance
            if not self.importance:
                self.importance = dict(zerolike_params_dict(strategy.model))

    def after_training_exp(self, strategy, **kwargs):
        if strategy.EpochClock.train_exp_epochs == strategy.EpochClock.max_epochs:
            print("update")
            self.params = dict(copy_params_dict(strategy.model))

            # Check if previous importance is available
            if not self.importance:
                raise ValueError("Importance is not available")

            # Get importance
            curr_importance = self._get_importance(strategy.model,strategy.experience.dataset, strategy.train_mb_size, strategy.device)

            # Update importance
            for name in self.importance.keys():
                self.importance[name] = (
                    self.alpha * self.importance[name]
                    + (1 - self.alpha) * curr_importance[name]
                )