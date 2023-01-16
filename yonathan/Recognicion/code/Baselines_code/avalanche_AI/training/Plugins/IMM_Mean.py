"""
IMM Mean plugin.
Penalties by L2 loss the old & current weights.
Then merges by mean the weights.
"""
import argparse
import os
import sys
import torch.nn as nn
import torch

from avalanche.training.templates.supervised import SupervisedTemplate as Regularization_strategy

from Baselines_code.baselines_utils import compute_quadratic_loss
from Baselines_code.baselines_utils import set_model

from training.Data.Data_params import RegType
from Baselines_code.avalanche_AI.training.Plugins.plugins_base import Base_plugin
from training.Data.Structs import inputs_to_struct
from typing import Union

sys.path.append(r'/')


class IMM_Mean(Base_plugin):
    """
    EWC plugin.
    Stores for each parameter its importance.
    """

    def __init__(self, opts: argparse, prev_checkpoint: Union[None, dict] = None):
        """
        Args:
            opts: The model opts.
            prev_checkpoint: A pretrained model
        """
        super(IMM_Mean, self).__init__(opts=opts, prev_checkpoint=prev_checkpoint, reg_type=RegType.IMM_Mean)
        # Supporting pretrained model.
        self.num_exp = 1 if prev_checkpoint is not None else 0
        self.importances = {name: 1 for name, _ in self.prev_model.feature_extractor.named_parameters()}

    def penalty(self, model: nn.Module, mb_x: inputs_to_struct, **kwargs):
        """
        Return the L2 penalty.
        Args:
            model: The current model.
            mb_x: The input.
            **kwargs: optional kwargs.

        Returns:

        """
        return compute_quadratic_loss(model, self.prev_model, importance=self.importances,
                                      device=self.device)

    def after_training_exp(self, strategy: Regularization_strategy, **kwargs) -> None:
        """
        Compute importance of parameters after each experience.
        Args:
            strategy: The strategy.

        """
        exp_counter = strategy.clock.train_exp_counter + self.num_exp
        # Current model state.
        current_model_state = strategy.model.feature_extractor.state_dict()
        # Previous model state.
        prev_model_state = self.prev_model.feature_extractor.state_dict()
        # Update the current and previous model.
        for name, param in strategy.model.feature_extractor.named_parameters():
            current_model_state[name] = (current_model_state[name] + prev_model_state[name] * exp_counter) / (
                    exp_counter + 1)
        # Set the new weights into the model.
        set_model(strategy.model.feature_extractor, current_model_state)
        model_path_curr = os.path.join(strategy.Model_folder, 'Merged_model')
        strategy.model.state_dict()['IMM_Mean_state_dict'] = self.state_dict(strategy)
        # Save the model.
        torch.save(strategy.model.state_dict(), model_path_curr)

    def state_dict(self, strategy: Regularization_strategy):
        """
        Args:
            strategy: The strategy.

        Returns:

        """
        return self.prev_model.feature_extractor.state_dict(), strategy.model.feature_extractor.state_dict()
