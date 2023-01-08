import copy
import sys
from torch.utils.data import Dataset
from training.Utils import preprocess
from torch.utils.data import DataLoader
import torch
from avalanche.models.utils import avalanche_forward
from avalanche.training.utils import copy_params_dict, zerolike_params_dict
from avalanche.training.plugins import EWCPlugin
import argparse
import torch.nn as nn
from typing import Callable
from torch.optim import optimizer
from Baselines_code.baselines_utils import set_model
from avalanche.training.templates.supervised import SupervisedTemplate

sys.path.append(r'/')

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin


class MyIMM_Mode_Plugin(SupervisedPlugin):
    """
    EWC plugin.
    Stores for each parameter its importance.
    """

    def __init__(self, parser: argparse, prev_model=None, old_dataset:Dataset = None):
        """
        Args:
            parser: The model parser.
            mode: The training mode.
            keep_importance_data: Whether to keep the importance Data_Creation.
            prev_model: A pretrained model
            prev_data: The old dataset.
        """
        super().__init__()
        self.prev_model = copy.deepcopy(prev_model)  # The previous model.
        self.parser = parser  # The model opts.
        self.num_exp = 0  # The number of exp trained so far.
        self.inputs_to_struct = parser.inputs_to_struct  # The inputs to struct method.
        self.outs_to_struct = parser.outs_to_struct  # The outputs to struct method
        self.imm_mode_lambda = parser.imm_mode_lambda
        self.criterion = parser.criterion
        self.device = parser.device
        self.bs = parser.bs
        # Supporting pretrained model.
        if prev_model is not None:
            # Update importance and old params to begin with EWC training.
          #
            #     self.saved_params[0] = dict(copy_params_dict(prev_model.feature_extractor))  # Copy the old parameters.
            self.num_exp = 1  #
            dataloader = DataLoader(old_dataset, batch_size=self.bs)  # The dataloader.
            self.fisher = self.compute_importances(prev_model, self.criterion, dataloader,
                           self.parser.device,self.parser.bs)
            print('Done computing Importances')

    def after_training_exp(self, strategy: SupervisedTemplate) -> None:
        """
        Compute importance of parameters after each experience.
        Args:
            strategy: The strategy.

        """
        exp_counter = strategy.clock.train_exp_counter + self.num_exp
        fisher_new = self.compute_importances(strategy.model, strategy._criterion , strategy.dataloader,self.device, self.bs)
        for (n, curr_param), (_, prev_param) in zip(strategy.model.feature_extractor.named_parameters(), self.prev_model.feature_extractor.named_parameters()):
            curr_param = fisher_new[n] * curr_param + self.fisher[n] * prev_param
            self.fisher[n] += fisher_new[n]
            curr_param /= (self.fisher[n] == 0).float() + self.fisher[n]

        model_path_curr = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/Emnist/Baselines/'
        torch.save(strategy.model.state_dict(), model_path_curr)

    def before_backward(self, strategy: SupervisedTemplate) -> None:
        """
        Compute EWC penalty and add it to the loss.
        Args:
            strategy: The strategy.

        """
        exp_counter = strategy.clock.train_exp_counter + self.num_exp
        if exp_counter == 0 or self.imm_mode_lambda == 0.0:
            return
        penalty = torch.tensor(0).float().to(strategy.device)

        for (_, param), (_, param_old) in zip(strategy.model.feature_extractor.named_parameters(),
                                              self.prev_model.feature_extractor.named_parameters()):
            penalty += torch.sum((param - param_old).pow(2))
            # Update the new loss.
        print(penalty)
        strategy.loss += self.imm_mode_lambda * penalty

    def compute_importances(self, model: nn.Module, criterion: Callable, dataloader : Dataset,
                            device: str, batch_size: int) -> dict:
        """
        Compute EWC importance matrix for each parameter
        Args:
            model: The model we compute its coefficients.
            criterion: The loss criterion.
            optimizer: The optimizer.
            dataset: The dataset.
            device: The device.
            batch_size: The batch size.

        Returns: The importance coefficients.

        """
        model.eval()  # Move to evaluation mode.
        importances = zerolike_params_dict(model.feature_extractor)  # Make empty coefficients.

        for i, batch in enumerate(dataloader):  # Iterating over the dataloader.
            x = preprocess(batch, device)  # Omit the ids and move to the device.
            x = self.inputs_to_struct(x)  # Make a struct.
            model.zero_grad()  # Reset grads.
            out = avalanche_forward(model, x, task_labels=None)  # Compute output.
            out = self.outs_to_struct(out)  # Make a struct.
            loss = criterion(self.parser, x, out)  # Compute the loss.
            loss.backward()  # Compute grads.
            for (k1, p), (k2, imp) in zip(model.feature_extractor.named_parameters(),
                                          importances):  # Iterating over the feature weights.
                assert k1 == k2
                if p.grad is not None:
                    # Adding the grad**2.
                    imp += p.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp in importances:
            imp /= float(len(dataloader))
        # Make dictionary.
        importances = dict(importances)
        return importances