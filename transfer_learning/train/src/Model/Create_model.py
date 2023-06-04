"""
Creating the desired Model according to the Model layer_type
"""
from ..data.Enums import Model_type, TrainingFlag
from ..Modules.Model_types.ResNet.ResNet import *

import argparse
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
from torchvision.models.efficientnet import EfficientNet_B0_Weights, efficientnet_b0 as efficient

from ..Modules.Model_types.EfficientNet.EfficientNet import efficientnet_b0
from ..Modules.Model_types.MobileNet.MobileNetV2 import MobileNetV2
from ..Modules.Model_types.MobileNet.MobileNetV3 import mobilenet_v3_large


# TODO- ADD ASSERTS.

def create_model(opts: argparse, model_type: Model_type) -> nn.Module:
    """
    Creating Model.
    Args:
        opts: The Model opts.
        model_type: The Model layer_type.

    Returns: The Model.

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
        assert opts.training_flag is TrainingFlag.LWF or opts.training_flag is TrainingFlag.Full_Model
        return mobilenet_v3_large(opts=opts)
