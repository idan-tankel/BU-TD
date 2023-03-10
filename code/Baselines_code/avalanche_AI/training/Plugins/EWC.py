"""
Elastic Weight consolidation plugin.
Saved for each parameter its importance
And regularizes by quadratic loss.
"""
import argparse
import sys
from typing import Optional

import torch
import torch.nn as nn
from avalanche.training.templates.supervised import SupervisedTemplate as Regularization_strategy
from torch import Tensor
from torch.utils.data import DataLoader

from Baselines_code.avalanche_AI.training.Plugins.plugins_base import Base_plugin
from Baselines_code.baselines_utils import compute_quadratic_loss, compute_fisher_information_matrix, RegType

sys.path.append(r'/')


class EWC(Base_plugin):
    """
    EWC plugin.
    Stores for each parameter its importance.
    """

    def __init__(self, opts: argparse, prev_checkpoint: Optional[dict] = None,
                 load_from: Optional[str] = None):
        """
        Args:
            opts: The model opts.
            prev_checkpoint: A pretrained model
        """
        super(EWC, self).__init__(opts=opts, prev_checkpoint=prev_checkpoint, reg_type=RegType.EWC)
        dataloader = self.Get_old_dl()  # The old data-set for computing the coefficients.
        # Supporting pretrained model.
        if prev_checkpoint is not None and dataloader is not None and self.reg_factor != 0.0:
            # Update importance and old params to begin with EWC training.
            self.num_exp = 1  #
            if load_from is not None:
                model_meta_data = torch.load(load_from)
                try:
                    self.importances = model_meta_data['EWC_importances']
                    print("Loaded existing importances")
                except KeyError:
                    print('Computing Importances')
                    self.importances = compute_fisher_information_matrix(self.opts, self.prev_model, opts.criterion,
                                                                         dataloader,
                                                                         opts.device)
                    model_meta_data['EWC_importances'] = self.importances
                    torch.save(model_meta_data, load_from)
                    print("Computed EWC importances once for all!")

    def after_training_exp(self, strategy: Regularization_strategy, **kwargs) -> None:
        """
        Compute importance of parameters after each experience.
        Args:
            strategy: The strategy.

        """
        dataloader = DataLoader(strategy.experience.dataset, batch_size=self.opts.bs)  # The dataloader.
        self.importances = compute_fisher_information_matrix(
            opts=self.opts,
            model=strategy.model,
            criterion=strategy.criterion,
            dataloader=dataloader,
            device=strategy.device,
        )
        self.prev_model = strategy.model
        print("Finished here")

    def penalty(self, model: nn.Module, mb_x: list[Tensor], **kwargs) -> Tensor.float:
        """

        Args:
            model: The model.
            mb_x: The input.
            **kwargs:

        Returns:

        """
        return compute_quadratic_loss(model, self.prev_model, importance=self.importances,
                                      device=self.device)

    def state_dict(self, strategy: Regularization_strategy):
        """
        Args:
            strategy:

        Returns:

        """
        return self.importances
