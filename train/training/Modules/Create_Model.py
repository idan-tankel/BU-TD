"""
Here we create the model type according to model type.
"""
from ..Data.Enums import Model_Type
import argparse
from typing import Union
import torch.nn as nn

from ..Modules.BU_TD.Models import BUTDModel

from ..Modules.classification_Models.Models import ResNet14, ResNet18, ResNet20, ResNet32, ResNet34, ResNet50, \
    ResNet101, SimpleMLP


def create_model(opts: argparse, model_type: Model_Type) -> nn.Module:
    """
    Creating model.
    Args:
        opts: The model opts.
        model_type: The model type.

    Returns: The model.

    """

    if model_type is Model_Type.ResNet14:
        return ResNet14(opts)

    if model_type is Model_Type.ResNet18:
        return ResNet18(opts)
    if model_type is Model_Type.ResNet20:
        return ResNet20(opts)

    elif model_type is Model_Type.ResNet32:
        return ResNet32(opts)

    elif model_type is Model_Type.ResNet34:
        return ResNet34(opts)

    elif model_type is Model_Type.ResNet50:
        return ResNet50(opts)

    elif model_type is Model_Type.ResNet101:
        return ResNet101(opts)

    elif model_type is Model_Type.MLP:
        return SimpleMLP(opts=opts)

    elif model_type is Model_Type.BUTD:
        return BUTDModel(opts=opts)
            