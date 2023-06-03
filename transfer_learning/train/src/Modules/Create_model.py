"""
Creating the desired model according to the model layer_type
"""
from ..data.Enums import Model_type
from ..Modules.ResNet.ResNet import *

import argparse
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
from torchvision.models.efficientnet import EfficientNet_B0_Weights, efficientnet_b0 as efficient

from ..Modules.EfficientNet.EfficientNet import efficientnet_b0
from ..Modules.MobileNet.MobileNetV2 import MobileNetV2
from ..Modules.MobileNet.MobileNetV3 import mobilenet_v3_large


def create_model(opts: argparse, model_type: Model_type) -> nn.Module:
    """
    Creating model.
    Args:
        opts: The model opts.
        model_type: The model layer_type.

    Returns: The model.

    """

    if model_type is Model_type.ResNet14:
        return ResNet14(opts)

    if model_type is Model_type.ResNet18:
        return ResNet18(opts)
    if model_type is Model_type.ResNet20:
        return ResNet20(opts)

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

    elif model_type is Model_type.VIT:
        return vit_b_16(ViT_B_16_Weights)

    elif model_type is Model_type.EfficientNet:
        return efficientnet_b0(opts=opts)

    elif model_type is Model_type.EfficientNetOriginal:
        return efficient(EfficientNet_B0_Weights)

    elif model_type is Model_type.MobileNetV2:
        return MobileNetV2(opts=opts)
    elif model_type is Model_type.MobileNetV3:
        return mobilenet_v3_large(opts=opts)
