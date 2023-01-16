"""
MAS plugin.
Very similar to EWC
Just, with another coefficients
computation rule.
"""
import argparse
import sys
from typing import Dict, Union

import torch
from training.Data.Structs import inputs_to_struct
import torch.nn as nn
from avalanche.training.templates.supervised import SupervisedTemplate as Regularization_strategy
from torch.utils.data import DataLoader

from Baselines_code.baselines_utils import compute_quadratic_loss, compute_fisher_information_matrix, Norm
from training.Data.Data_params import RegType
from Baselines_code.avalanche_AI.training.Plugins.plugins_base import Base_plugin

sys.path.append(r'/')


# TODO -ADD SUCH DESCRIPTION FOR ALL PLUGINS.
class MAS(Base_plugin):
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

    def __init__(self, opts: argparse, prev_checkpoint: Union[dict, None] = None, load_from=None):
        """
        Args:
            opts: The model options.
            prev_checkpoint: The previous model.

        """

        # Init super class
        super(MAS, self).__init__(opts=opts, prev_checkpoint=prev_checkpoint, reg_type=RegType.MAS)
        # Regularization Parameters and Importances parameters.
        self.alpha = opts.mas_alpha
        # Model parameters
        self.importances: Union[Dict, None] = None  # The parameters importances.
        # If we have previous model we save it and compute its importances.
        if prev_checkpoint is not None:
            self.num_exp = 1
            dataloader = self.Get_old_dl()  # The old data-set for computing the coefficients.
            # Compute the importances.
            if load_from is not None:
                model_meta_data = torch.load(load_from)
                try:
                    self.importances = model_meta_data['MAS_importances']
                    print("Load computed importances.")
                except KeyError:
                    print("Computing Importances")
                    self.importances = compute_fisher_information_matrix(opts=opts, model=self.prev_model, norm=1,
                                                                         criterion=Norm, dataloader=dataloader,
                                                                         device='cuda')
                    model_meta_data['MAS_importances'] = self.importances
                    torch.save(model_meta_data, load_from)
                    print("Computed MAS importances once for all!")
            # The parameters we want to regularize are only the backbone parameters.
            # Update the number of trained experiences.

    def penalty(self, model: nn.Module, mb_x: inputs_to_struct, **kwargs):
        """

        Args:
            model: The current model.
            mb_x: The input.
            **kwargs: Possible args.

        Returns:

        """
        return compute_quadratic_loss(model, self.prev_model, importance=self.importances,
                                      device=self.device)

    def after_training_exp(self, strategy: Regularization_strategy, **kwargs) -> None:
        """
        Update the Importances after the experience is finished.
        Args:
            strategy: The strategy.
            **kwargs: Possible args.

        """
        print("update")
        # Get importance
        dataloader = DataLoader(strategy.experience.dataset, batch_size=self.opts.bs)  # The dataloader.
        # Compute the importances.
        curr_importances = compute_fisher_information_matrix(opts=self.opts, model=self.prev_model, norm=1,
                                                             criterion=Norm, dataloader=dataloader,
                                                             device='cuda')
        # Update importance
        for name in self.importances.keys():
            self.importances[name] = (self.alpha * self.importances[name] + (1 - self.alpha) * curr_importances[name])

    def state_dict(self, strategy: Regularization_strategy):
        """
        Args:
            strategy: The strategy.

        Returns:

        """
        return self.importances
