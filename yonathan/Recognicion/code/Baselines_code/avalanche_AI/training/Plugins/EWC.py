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
from avalanche.training.templates.supervised import SupervisedTemplate

sys.path.append(r'/')


class MyEWCPlugin(EWCPlugin):
    def __init__(self, parser: argparse, mode="separate", decay_factor=None, keep_importance_data=False,
                 prev_model=None, old_dataset=None):
        """
        Args:
            parser: The model parser.
            mode: The training mode.
            decay_factor:
            keep_importance_data: Whether to keep the importance data.
            prev_model: A pretrained model
            old_dataset: The old dataset.
        """
        super().__init__(ewc_lambda=parser.EWC_lambda, mode=mode, decay_factor=decay_factor,
                         keep_importance_data=keep_importance_data)
        self.old_dataset = old_dataset
        self.prev_model = prev_model
        self.parser = parser
        self.num_exp = 0
        self.inputs_to_struct = parser.inputs_to_struct
        self.outs_to_struct = parser.outs_to_struct
        # Supporting pretrained model.
        if prev_model is not None and old_dataset is not None:
            # Update importance and old params to begin with EWC training.
            print("Computing Importances")
            importances = self.compute_importances(prev_model, parser.criterion, parser.optimizer, old_dataset,
                                                   parser.device, parser.train_mb_size)
            self.update_importances(importances, 0)  # The first task.
            print("Done computing Importances")
            self.saved_params[0] = dict(copy_params_dict(prev_model.feature))  # Change to the excluded params.
            self.num_exp = 1

    def compute_importances(self, model: nn.Module, criterion: Callable, optimizer: optimizer, dataset: Dataset,
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
        model.eval()
        importances = zerolike_params_dict(model.feature)  # Make empty coefficients.
        dataloader = DataLoader(dataset, batch_size=batch_size)  # The dataloader.
        for i, batch in enumerate(dataloader):  # Iterating over the dataloader.
            x = preprocess(batch[:-1], device)  # Omit the ids and move to the device.
            x = self.inputs_to_struct(x)  # Make a struct.
            optimizer.zero_grad()  # Reset grads.
            out = avalanche_forward(model, x, task_labels=None)  # Compute output.
            out = self.outs_to_struct(out)  # Make a struct.
            loss = criterion(self.parser, x, out)  # Compute the loss.
            loss.backward()  # Compute grads.
            for (k1, p), (k2, imp) in zip(model.feature.named_parameters(),
                                          importances):  # Iterating over the feature weights.
                assert k1 == k2
                if p.grad is not None:
                    imp += p.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp in importances:
            imp /= float(len(dataloader))
        importances = dict(importances)
        return importances

    def after_training_exp(self, strategy: SupervisedTemplate) -> None:
        """
        Compute importance of parameters after each experience.
        Args:
            strategy: The strategy.

        """
        exp_counter = strategy.clock.train_exp_counter + self.num_exp
        importances = self.compute_importances(
            strategy.model,
            strategy._criterion,
            strategy.optimizer,
            strategy.experience.dataset,
            strategy.device,
            strategy.train_mb_size,
        )
        # Update importance.
        self.update_importances(importances, exp_counter)
        self.saved_params[exp_counter] = copy_params_dict(strategy.model.feature)
        # clear previous parameter values
        if exp_counter > 0 and (not self.keep_importance_data):
            del self.saved_params[exp_counter - 1]

    def before_backward(self, strategy: SupervisedTemplate) -> None:
        """
        Compute EWC penalty and add it to the loss.
        Args:
            strategy: The strategy.

        """
        exp_counter = strategy.clock.train_exp_counter + self.num_exp
        if exp_counter == 0 or self.ewc_lambda == 0.0:
            return
        penalty = torch.tensor(0).float().to(strategy.device)

        Cur_params = dict(strategy.model.feature.named_parameters())
        for name in self.importances[0].keys():
            saved_param = self.saved_params[0][name]  # previous weight.
            imp = self.importances[0][name]  # Current weight.
            cur_param = Cur_params[name]
            # Add the difference to the loss.
            penalty += (imp * (cur_param - saved_param).pow(2)).sum()

        strategy.loss += self.ewc_lambda * penalty
