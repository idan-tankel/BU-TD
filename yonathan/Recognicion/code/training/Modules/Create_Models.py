import argparse

import torch.nn as nn

from training.Modules.Models import ResNet, BUTDModel


def create_model(opts: argparse) -> nn.Module:
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
