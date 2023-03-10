"""
LFL plugin.
Stores the old model and penalties
The L2 feature distance.
"""
import argparse
import sys
from typing import Union, Optional

import torch
import torch.nn as nn
from torch import Tensor

from Baselines_code.avalanche_AI.training.Plugins.plugins_base import Base_plugin
from training.Data.Data_params import RegType
from training.Data.Structs import inputs_to_struct

sys.path.append(r'/')


# Code credit from Avalanche-AI

class LFL(Base_plugin):
    """Less-Forgetful Learning (LFL) Plugin.
    LFL satisfies two properties to mitigate catastrophic forgetting.
    1) To keep the decision boundaries unchanged
    2) The feature space should not change much on target(new) Data_Creation
    LFL uses euclidean loss between features from current and previous version
    of model as regularization to maintain the feature space and avoid
    catastrophic forgetting.
    Refer paper https://arxiv.org/pdf/1607.00122.pdf for more details
    This plugin does not use task identities.
    """

    def __init__(self, opts: argparse, prev_checkpoint: Optional[dict]):
        """
        Create LFL plugin, if prev_model is not None, we copy its parameters.
        Args:
            opts: The model options.
            prev_checkpoint: The prev model.
        """
        super(LFL, self).__init__(opts=opts, prev_checkpoint=prev_checkpoint, reg_type=RegType.LFL)

    def compute_features(self, model: nn.Module, x: inputs_to_struct) -> tuple[Tensor, Tensor]:
        """
        Compute features from prev model and current model
        Args:
            model: The current model.
            x: The input to the models.

        Returns: The old, new features.

        """
        model.eval()  # Move to eval mode.
        self.prev_model.eval()  # Move to eval model.
        features = model.forward_and_out_to_struct(x).features  # New features.
        prev_features = self.prev_model.forward_and_out_to_struct(x).features  # Old features.
        return features, prev_features

    def penalty(self, model: nn.Module, x: inputs_to_struct, **kwargs) -> torch.float:
        """
        Compute weighted euclidean loss
        Args:
            x: The input to the model.
            model: The current model.

        Returns: The weighted MSE loss.

        """
        # The previous, current features.
        features, prev_features = self.compute_features(model, x)
        # Compute distance loss.
        dist_loss = torch.nn.functional.mse_loss(features, prev_features)
        return self.reg_factor * dist_loss
