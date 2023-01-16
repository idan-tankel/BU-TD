"""
Here we create the model type according to model type.
"""
import argparse

from training.Modules.Models import ResNet, BUTDModel

from typing import Union


def create_model(opts: argparse) -> Union[ResNet, BUTDModel]:
    """
    Create and return a model according to the options.
    Args:
        opts: The model opts.

    Returns: A model.

    """
    if opts.model_type is BUTDModel:
        return BUTDModel(opts=opts).cuda()
    else:
        return ResNet(opts=opts).cuda()
