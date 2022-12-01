import copy
import sys
from typing import Dict, Union
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import torch
from avalanche.training.utils import copy_params_dict, zerolike_params_dict
from avalanche.training.plugins import MASPlugin
from torch.utils.data import Dataset
from training.Utils import preprocess
from avalanche.training.templates.supervised import SupervisedTemplate

sys.path.append(r'/')


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

    def __init__(self, parser: argparse, prev_model: Union[nn.Module, None] = None):
        """
        Args:
            parser: The parser options.
            prev_model: The previous model.
        """

        # Init super class
        super().__init__()

        # Regularization Parameters
        self.batch_size = parser.bs
        self.device = parser.device
        self.parser = parser
        self._lambda = parser.MAS_lambda
        self.alpha = parser.mas_alpha
        # Model parameters
        self.params: Union[Dict, None] = None
        self.importance: Union[Dict, None] = None
        self.num_exp = 0
        self.inputs_to_struct = parser.inputs_to_struct
        if prev_model is not None:
            self.prev_model = copy.deepcopy(prev_model)
            self.prev_data = parser.prev_data
            print("Computing Importances")
            self.importance = self._get_importance(self.prev_model, self.prev_data, self.batch_size, parser.device)
            print("Done computing Importances")
            self.params = self.params = dict(copy_params_dict(self.prev_model.feature))
            self.num_exp = 1

    def _get_importance(self, model: nn.Module, dataset: Dataset, train_mb_size: int, device: Union['cuda', 'cpu']):

        # Initialize importance matrix
        importance = dict(zerolike_params_dict(model.feature))
        # Do forward and backward pass to accumulate L2-loss gradients
        model.train()
        dataloader = DataLoader(dataset, batch_size=train_mb_size, )
        # Progress bar
        for _, batch in enumerate(dataloader):
            # Get batch
            # Move batch to device
            batch = preprocess(batch[:-1], device)
            batch = self.inputs_to_struct(batch)
            # Forward pass
            model.zero_grad()
            # Forward pass
            out = model.forward_and_out_to_struct(batch).classifier
            # Average L2-Norm of the output
            loss = torch.norm(out, p="fro", dim=1).mean()
            loss.backward()
            # Accumulate importance
            for name, param in model.feature.named_parameters():
                if param.requires_grad:
                    if param.grad is not None:
                        importance[name] += param.grad.abs()

        # Normalize importance
        importance = {name: importance[name] / len(dataloader) for name in importance.keys()}
        return importance

    def before_backward(self, strategy: SupervisedTemplate, **kwargs):
        # Check if the task is not the first
        exp_counter = strategy.clock.train_exp_counter + self.num_exp
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
        for name, param in strategy.model.feature.named_parameters():
            if name in self.importance.keys():
                loss_reg += torch.sum(self.importance[name] * (param - self.params[name]).pow(2))

        # Update loss
        strategy.loss += self._lambda * loss_reg

    def before_training(self, strategy: SupervisedTemplate, **kwargs):
        # Parameters before the first task starts

        if self.num_exp == 0:
            if not self.params:
                self.params = dict(copy_params_dict(strategy.model))

            # Initialize Fisher information weight importance
            if not self.importance:
                self.importance = dict(zerolike_params_dict(strategy.model))

    def after_training_exp(self, strategy, **kwargs):
        print("update")
        self.params = dict(copy_params_dict(strategy.model))

        # Check if previous importance is available
        if not self.importance:
            raise ValueError("Importance is not available")

        # Get importance
        curr_importance = self._get_importance(strategy.model, strategy.experience.dataset, strategy.train_mb_size,
                                               strategy.device)

        # Update importance
        for name in self.importance.keys():
            self.importance[name] = (self.alpha * self.importance[name] + (1 - self.alpha) * curr_importance[name])
