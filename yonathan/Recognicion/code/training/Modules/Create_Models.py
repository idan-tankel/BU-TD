import argparse

import torch.nn as nn

from training.Modules.Models import ResNet, BUTDModel


def create_model(parser: argparse) -> nn.Module:
    """
    Create and return a model according to the options.
    Args:
        parser: The parser.

    Returns: A model.

    """
    if parser.model_type is BUTDModel:
        return BUTDModel(parser).cuda()
    else:
        return ResNet(parser).cuda()
