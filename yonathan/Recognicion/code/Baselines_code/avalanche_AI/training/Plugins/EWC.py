import argparse
import copy
import sys
from typing import Union

import torch
import torch.nn as nn
from avalanche.training.plugins import EWCPlugin
from avalanche.training.templates.supervised import SupervisedTemplate as Regularization_strategy
from avalanche.training.utils import copy_params_dict
from torch.utils.data import DataLoader, Dataset

from Baselines_code.baselines_utils import compute_quadratic_loss, compute_fisher_information_matrix

sys.path.append(r'/')


class EWC(EWCPlugin):
    """
    EWC plugin.
    Stores for each parameter its importance.
    """

    def __init__(self, parser: argparse, prev_model: Union[None, nn.Module] = None,
                 old_dataset: Union[None, Dataset] = None, load_from=None):
        """
        Args:
            parser: The model parser.
            prev_model: A pretrained model
            old_dataset: The old dataset.
        """
        super().__init__(ewc_lambda=parser.EWC_lambda)
        self.old_dataset = old_dataset  # The old data-set for computing the coefficients.
        self.prev_model = copy.deepcopy(prev_model)  # The previous model.
        self.parser = parser  # The model opts.
        self.num_exp = 0  # The number of exp trained so far.
        self.inputs_to_struct = parser.inputs_to_struct  # The inputs to struct method.
        self.outs_to_struct = parser.outs_to_struct  # The outputs to struct method
        # Supporting pretrained model.
        if prev_model is not None and old_dataset is not None and self.ewc_lambda != 0.0:
            # Update importance and old params to begin with EWC training.
            print('Computing Importances')
            dataloader = DataLoader(old_dataset, batch_size=self.parser.bs)  # The dataloader.
            if load_from is not None:
                model_meta_data = torch.load(load_from)
                try:
                    self.importances = model_meta_data['EWC_importances']
                    print("Loaded existing importances")
                except KeyError:
                    self.importances = compute_fisher_information_matrix(self.parser, prev_model, parser.criterion,
                                                                         dataloader,
                                                                         parser.device)
                    model_meta_data['EWC_importances'] = self.importances
                    torch.save(model_meta_data, load_from)
                    print("Computed EWC importances once for all!")
         #   print('Done computing Importances')
            self.saved_params[0] = dict(copy_params_dict(prev_model.feature_extractor))  # Copy the old parameters.
            self.num_exp = 1  #

    def after_training_exp(self, strategy: Regularization_strategy, **kwargs) -> None:
        """
        Compute importance of parameters after each experience.
        Args:
            strategy: The strategy.

        """
        exp_counter = strategy.clock.train_exp_counter + self.num_exp
        dataloader = DataLoader(strategy.experience.dataset, batch_size=self.parser.bs)  # The dataloader.
        self.importances = compute_fisher_information_matrix(
            parser=self.parser,
            model=strategy.model,
            criterion=strategy._criterion,
            dataloader=dataloader,
            device=strategy.device,
        )
        # Update importance.
        # Update the new 'old' weights.
        self.saved_params[exp_counter] = copy_params_dict(strategy.model.feature_extractor)
        # clear previous parameter values
        if exp_counter > 0 and (not self.keep_importance_data):
            del self.saved_params[exp_counter - 1]

    def before_backward(self, strategy: Regularization_strategy, **kwargs) -> None:
        """
        Compute EWC penalty and add it to the loss.
        Args:
            strategy: The strategy.

        """
        exp_counter = strategy.clock.train_exp_counter + self.num_exp
        if exp_counter == 0 or self.ewc_lambda == 0.0:
            return

        penalty = compute_quadratic_loss(strategy.model, self.prev_model, importance=self.importances,
                                         device=strategy.device)
        # print(penalty)
        loss = strategy.loss
        convex_loss = self.ewc_lambda * penalty + (1 - self.ewc_lambda) * loss
        strategy.loss = convex_loss
