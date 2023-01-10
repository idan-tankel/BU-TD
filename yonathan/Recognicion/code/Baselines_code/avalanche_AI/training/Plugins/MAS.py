import argparse
import copy
import sys
from typing import Dict, Union

import torch
import torch.nn as nn
from avalanche.training.plugins import MASPlugin
from avalanche.training.templates.supervised import SupervisedTemplate as Regularization_strategy
from avalanche.training.utils import copy_params_dict, zerolike_params_dict
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from Baselines_code.baselines_utils import compute_quadratic_loss, compute_fisher_information_matrix

sys.path.append(r'/')


class MAS(MASPlugin):
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

    def __init__(self, parser: argparse, prev_model: Union[nn.Module, None] = None,
                 prev_data: Union[Dataset, None] = None, load_from=None):
        """
        Args:
            parser: The parser options.
            prev_model: The previous model.
            prev_data: The previous data.
        """

        # Init super class
        super().__init__()

        # Regularization Parameters and Importances parameters.
        self.batch_size, self.device, self.parser, self.reg_factor, self.alpha = parser.bs, parser.device, parser, \
            parser.MAS_lambda, parser.mas_alpha
        # Model parameters
        self.params: Union[Dict, None] = None  # The parameters we want to regularize.
        self.importances: Union[Dict, None] = None  # The parameters importances.
        self.num_exp = 0  # No experiences trained so far.
        self.inputs_to_struct = parser.inputs_to_struct  # Input to struct.
        # If we have previous model we save it and compute its importances.
        if prev_model is not None:
            self.prev_model = copy.deepcopy(prev_model)  # Previous model.
            print("Computing Importances")

            def Norm(parser, x, out):
                return torch.norm(out.classifier, dim=1).pow(2).mean()

            dataloader = DataLoader(prev_data, batch_size=self.parser.bs)  # The dataloader.
            # Compute the importances.

            #
            if load_from is not None:
                model_meta_data = torch.load(load_from)
                try:
                    self.importances = model_meta_data['MAS_importances']
                except KeyError:
                    self.importances = compute_fisher_information_matrix(parser=parser, model=self.prev_model, norm=1,
                                                                         criterion=Norm, dataloader=dataloader,
                                                                         device='cuda')
                    model_meta_data['MAS_importances'] = self.importances
                    torch.save(model_meta_data, load_from)
                    print("Computed MAS importances once for all!")
            #
            print("Done computing Importances")
            # The parameters we want to regularize are only the backbone parameters.
            self.params = dict(copy_params_dict(self.prev_model.feature_extractor))
            # Update the number of trained experiences.
            self.num_exp = 1

    def before_backward(self, strategy: Regularization_strategy, **kwargs) -> None:
        """
        Add the MAS loss to the classification loss.
        Args:
            strategy: The strategy.
            **kwargs:

        Returns:

        """
        # Check if the task is not the first
        exp_counter = strategy.clock.train_exp_counter + self.num_exp
        if exp_counter == 0 or self.reg_factor == 0.0:
            return

        penalty = compute_quadratic_loss(strategy.model, self.prev_model, importance=self.importances,
                                         device=strategy.device)
        strategy.loss += self.reg_factor * penalty

    def before_training(self, strategy: Regularization_strategy, **kwargs) -> None:
        """
        Before training initialize the parameters, importances.
        Args:
            strategy: The strategy.
            **kwargs:

        """
        # Parameters before the first task starts
        # If no tasks trained so far.
        if self.num_exp == 0:
            if not self.params:
                self.params = dict(copy_params_dict(strategy.model))

            # Initialize Fisher information weight importance
            if not self.importances:
                self.importances = dict(zerolike_params_dict(strategy.model))

    def after_training_exp(self, strategy: Regularization_strategy, **kwargs) -> None:
        """
        Update the Importances after the experience is finished.
        Args:
            strategy: The strategy.
            **kwargs:

        """
        print("update")
        self.params = dict(copy_params_dict(strategy.model))
        # Check if previous importance is available
        if not self.importances:
            raise ValueError("Importance is not available")

        # Get importance
        def Norm(parser, x, out):
            return torch.norm(out.classifier, dim=1).pow(2).mean()

        # Norm_one = lambda parser, x, out: torch.norm(out.classifier, dim=1).pow(2).mean()
        dataloader = DataLoader(strategy.experience.dataset, batch_size=self.parser.bs)  # The dataloader.
        # Compute the importances.
        curr_importances = compute_fisher_information_matrix(parser=self.parser, model=self.prev_model, norm=1,
                                                             criterion=Norm, dataloader=dataloader,
                                                             device='cuda')
        # Update importance
        for name in self.importances.keys():
            self.importances[name] = (self.alpha * self.importances[name] + (1 - self.alpha) * curr_importances[name])
