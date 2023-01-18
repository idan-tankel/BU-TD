"""
Here we create the model_test type according to model_test type.
"""
import argparse
from typing import Union

from training.Modules.Models import ResNet, BUTDModel


def create_model(opts: argparse) -> Union[ResNet, BUTDModel]:
    """
    Create and return a model_test according to the options.
    Args:
        opts: The model_test model_opts.

    Returns: A model_test.

    """
    if opts.model_type is BUTDModel:
        return BUTDModel(opts=opts).cuda()
    else:
        return ResNet(opts=opts).cuda()
