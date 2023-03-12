"""
Creating the desired model according to the model type
"""
from src.data.Enums import Model_type
from src.Modules.models import *
from typing import Union
import argparse


def create_model(opts: argparse, model_type: Model_type) -> Union[ResNet, SimpleMLP]:
    """
    Creating model.
    Args:
        opts: The model opts.
        model_type: The model type.

    Returns: The model.

    """
    if model_type is Model_type.ResNet18:
        return ResNet18(opts)

    elif model_type is Model_type.ResNet32:
        return ResNet32(opts)

    elif model_type is Model_type.ResNet34:
        return ResNet34(opts)

    elif model_type is Model_type.ResNet50:
        return ResNet50(opts)

    elif model_type is Model_type.ResNet101:
        return ResNet101(opts)

    elif model_type is Model_type.MLP:
        return SimpleMLP(opts=opts)
