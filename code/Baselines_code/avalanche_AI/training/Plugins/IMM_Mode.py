"""
IMM Mean plugin.
Similar to IMM Mean but
Merges differently the models and
uses the EWC importances.
"""
import argparse
import sys
from typing import Union, Optional

import torch
import torch.nn as nn
from avalanche.training.templates.supervised import SupervisedTemplate
from avalanche.training.templates.supervised import SupervisedTemplate as Regularization_strategy

from Baselines_code.avalanche_AI.training.Plugins.plugins_base import Base_plugin
from Baselines_code.baselines_utils import compute_quadratic_loss, compute_fisher_information_matrix
from training.Data.Data_params import RegType
from training.Data.Structs import inputs_to_struct

sys.path.append(r'/')


class MyIMM_Mode(Base_plugin):
    """
    EWC plugin.
    Stores for each parameter its importance.
    """

    def __init__(self, opts: argparse, prev_checkpoint: Optional[dict], load_from=None):
        """
        Args:
            opts: The model opts.
            prev_checkpoint: A pretrained model
            load_from: Path we load from
        """
        super(MyIMM_Mode, self).__init__(opts=opts, prev_checkpoint=prev_checkpoint, reg_type=RegType.IMM_Mode)
        self.fisher_new = None
        self.load_from = load_from
        # Supporting pretrained model.
        if prev_checkpoint is not None:
            self.num_exp = 1  #
            dataloader = self.Get_old_dl()  # The old data-set for computing the coefficients.
            model_meta_data = torch.load(load_from)
            try:
                self.importances = model_meta_data['EWC_importances']
            except KeyError:
                self.fisher = compute_fisher_information_matrix(self.opts, self.prev_model, opts.criterion, dataloader,
                                                                self.opts.device, self.opts.bs)
                model_meta_data['EWC_importances'] = self.importances
                torch.save(model_meta_data, load_from)
                print("Computed EWC importances once for all!")
            self.importances = {name: 1 for name, _ in self.prev_model.feature_extractor.named_parameters()}

    def after_training_exp(self, strategy: SupervisedTemplate, **kwargs) -> None:
        """
        Compute importance of parameters after each experience.
        Args:
            strategy: The strategy.

        """
        self.fisher_new = compute_fisher_information_matrix(self.opts, strategy.model, strategy.criterion,
                                                            strategy.dataloader, self.device, self.opts.bs)
        for (n, curr_param), (_, prev_param) in zip(strategy.model.feature_extractor.named_parameters(),
                                                    self.prev_model.feature_extractor.named_parameters()):
            curr_param = self.fisher_new[n] * curr_param + self.fisher[n] * prev_param
            self.fisher[n] += self.fisher_new[n]
            curr_param /= (self.fisher[n] == 0).float() + self.fisher[n]

        torch.save(strategy.model.state_dict(), strategy.checkpoint.dir_path)

    def state_dict(self, strategy: Regularization_strategy):
        """
        Args:
            strategy: The training strategy.

        Returns:

        """
        return (self.prev_model.feature_extractor.named_parameters(),
                strategy.model.feature_extractor.named_parameters(), self.fisher, self.fisher_new)

    def penalty(self, model: nn.Module, mb_x: inputs_to_struct, **kwargs):
        """

        Args:
            model: The model.
            mb_x: The input.
            **kwargs: The args.

        Returns:

        """
        return compute_quadratic_loss(model, self.prev_model, importance=self.importances,
                                      device=self.device)
