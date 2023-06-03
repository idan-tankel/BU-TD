"""
Different Models loading.
"""
from typing import Optional

from ..data.Enums import Model_type
import torch.nn as nn
import os
import torch


def load_model(results_dir: str, model_path: str):
    """
    Loads and returns the model checkpoint as a dictionary.
    Args:
        model_path: The path to the model.
        results_dir: Trained model dir.

    Returns: The loaded checkpoint.

    """
    model_path = os.path.join(results_dir, model_path)  # The path to the model.
    checkpoint = torch.load(model_path)  # Loading the saved data.
    return checkpoint['state_dict']


def Load_pretrained_resnet(model: nn.Module) -> None:
    """
    Load pretrained model.
    Args:
        model: The model.

    """
    checkpoint = {}
    state_dict = {}
    prefix = ''
    num_blocks = [5, 5, 5]
    for layer_id in range(len(num_blocks)):
        for block_id in range(num_blocks[layer_id]):
            for layer_type in ['conv1', 'conv2', 'conv3']:
                for param_type in ['weight']:
                    new_key = f'layers.{layer_id}.{block_id}.{layer_type}.{param_type}'
                    old_key = f'{prefix}layer{layer_id + 1}.{block_id}.{layer_type}.{param_type}'
                    try:
                        if new_key in model.state_dict().keys():
                            checkpoint[new_key] = state_dict[old_key]
                    except KeyError:
                        pass
            for layer_type in ['bn1', 'bn2', 'bn3']:
                for param_type in ['weight', 'bias']:
                    new_key = f'layers.{layer_id}.{block_id}.{layer_type}.norm.{param_type}'
                    old_key = f'{prefix}layer{layer_id + 1}.{block_id}.{layer_type}.{param_type}'
                    try:
                        if new_key in model.state_dict().keys():
                            checkpoint[new_key] = state_dict[old_key]
                    except KeyError:
                        pass

            for layer_type in ['downsample.conv1x1']:
                for param_type in ['weight']:
                    new_key = f'layers.{layer_id}.{block_id}.{layer_type}.{param_type}'
                    old_key = f'{prefix}layer{layer_id + 1}.{block_id}.downsample.0.{param_type}'
                    if new_key in model.state_dict().keys():
                        checkpoint[new_key] = state_dict[old_key]

            for layer_type in ['conv1', 'conv2', 'conv3', 'conv4']:
                new_key = f'layers.{layer_id}.{block_id}.modulated_{layer_type}.layer.weight'
                old_key = f'layers.{layer_id}.{block_id}.{layer_type}.weight'

                if layer_type == 'conv4':
                    old_key = f'layers.{layer_id}.{block_id}.downsample.conv1x1.weight'
                try:
                    checkpoint[new_key] = checkpoint[old_key]
                except KeyError:
                    pass

            for layer_type in ['downsample.norm.norm']:
                for param_type in ['weight', 'bias']:
                    new_key = f'layers.{layer_id}.{block_id}.{layer_type}.{param_type}'
                    old_key = f'{prefix}layer{layer_id + 1}.{block_id}.downsample.1.{param_type}'
                    if new_key in model.state_dict().keys():
                        checkpoint[new_key] = state_dict[old_key]

            checkpoint['conv1.weight'] = state_dict[f'{prefix}conv1.weight']
            checkpoint['bn1.norm.weight'] = state_dict[f'{prefix}bn1.weight']
            checkpoint['bn1.norm.bias'] = state_dict[f'{prefix}bn1.bias']

    for key, val in model.state_dict().items():
        if "running_mean" in key or 'running_var' in key or 'linear' in key or 'modulation' in key or 'mask' in key:
            checkpoint[key] = val

    model.load_state_dict(checkpoint)  # Loading the saved weights.


def Load_Pretrained_EfficientNet(model: nn.Module, model_type: Model_type) -> None:
    """
    Load EfficientNet.
    Args:
        model: The model.
        model_type: The model layer_type.

    Returns: None

    """
    num_layers = 7
    weights = model_type.weights().IMAGENET1K_V1
    old_state_dict = weights.get_state_dict(progress=True)
    current_state_dict = {}
    # current_state_dict = copy.deepcopy(old_state_dict)
    for name, param in model.state_dict().items():
        if "mask" in name or "running" in name or "classifier" in name:
            current_state_dict[name] = param

    for layer_id in range(num_layers):
        num_blocks = model.inverted_residual_setting[layer_id].num_layers

        for block_id in range(num_blocks):
            # Begin with conv.
            for bit in [0, 2]:
                try:
                    old_bit = 0 if bit == 0 else 3
                    old_key = f"features.{layer_id + 1}.{block_id}.block.{old_bit}.0" \
                              f".weight"
                    new_key = f'features.{layer_id + 1}.{block_id}.block.{old_bit}.conv.weight'
                    current_state_dict[new_key] = old_state_dict[old_key]
                except KeyError:
                    pass
            #
            try:
                new_key = f'features.{layer_id + 1}.{block_id}.block.1.conv.weight'
                old_key = f'features.{layer_id + 1}.{block_id}.block.1.0.weight'
                current_state_dict[new_key] = old_state_dict[old_key]
            except KeyError:
                pass
            try:
                new_key = f'features.{layer_id + 1}.{block_id}.block.1.bn.weight'
                old_key = f'features.{layer_id + 1}.{block_id}.block.1.1.weight'
                current_state_dict[new_key] = old_state_dict[old_key]
            except KeyError:
                pass
            try:
                new_key = f'features.{layer_id + 1}.{block_id}.block.1.bn.bias'
                old_key = f'features.{layer_id + 1}.{block_id}.block.1.1.bias'
                current_state_dict[new_key] = old_state_dict[old_key]
            except KeyError:
                pass
            try:
                new_key = f'features.{layer_id + 1}.{block_id}.block.2.bn.bias'
                old_key = f'features.{layer_id + 1}.{block_id}.block.2.1.bias'
                current_state_dict[new_key] = old_state_dict[old_key]
            except KeyError:
                pass
            try:
                new_key = f'features.{layer_id + 1}.{block_id}.block.2.bn.weight'
                old_key = f'features.{layer_id + 1}.{block_id}.block.2.1.weight'
                current_state_dict[new_key] = old_state_dict[old_key]
            except KeyError:
                pass
            #
            try:
                new_key = f'features.{layer_id + 1}.{block_id}.block.3.bn.bias'
                old_key = f'features.{layer_id + 1}.{block_id}.block.3.1.bias'
                current_state_dict[new_key] = old_state_dict[old_key]
            except KeyError:
                pass
            try:
                new_key = f'features.{layer_id + 1}.{block_id}.block.3.bn.weight'
                old_key = f'features.{layer_id + 1}.{block_id}.block.3.1.weight'
                current_state_dict[new_key] = old_state_dict[old_key]
            except KeyError:
                pass
            #
            # Continue to BN layers.
            old_key = f"features.{layer_id + 1}.{block_id}.block.0.1.weight"
            new_key = f'features.{layer_id + 1}.{block_id}.block.0.bn.weight'
            current_state_dict[new_key] = old_state_dict[old_key]
            old_key = f"features.{layer_id + 1}.{block_id}.block.0.1.bias"
            new_key = f'features.{layer_id + 1}.{block_id}.block.0.bn.bias'
            current_state_dict[new_key] = old_state_dict[old_key]
            # FC layer
            for bit in [1, 2]:
                for layer in ['weight', 'bias']:
                    old_key = f'features.{layer_id + 1}.{block_id}.block.2.fc{bit}.{layer}'
                    new_key = f'features.{layer_id + 1}.{block_id}.block.2.fc{bit}.{layer}'
                    try:
                        current_state_dict[new_key] = old_state_dict[old_key]
                    except KeyError:
                        pass

    current_state_dict['features.8.conv.weight'] = old_state_dict['features.8.0.weight']
    current_state_dict['features.8.bn.weight'] = old_state_dict['features.8.1.weight']
    current_state_dict['features.8.bn.bias'] = old_state_dict['features.8.1.bias']
    current_state_dict['features.0.conv.weight'] = old_state_dict['features.0.0.weight']
    current_state_dict['features.0.bn.weight'] = old_state_dict['features.0.1.weight']
    current_state_dict['features.0.bn.bias'] = old_state_dict['features.0.1.bias']
    current_state_dict['features.1.0.block.2.conv.weight'] = old_state_dict['features.1.0.block.2.0.weight']
    current_state_dict['features.1.0.block.1.fc1.weight'] = old_state_dict['features.1.0.block.1.fc1.weight']
    current_state_dict['features.1.0.block.1.fc1.bias'] = old_state_dict['features.1.0.block.1.fc1.bias']
    current_state_dict['features.1.0.block.1.fc2.weight'] = old_state_dict['features.1.0.block.1.fc2.weight']
    current_state_dict['features.1.0.block.1.fc2.bias'] = old_state_dict['features.1.0.block.1.fc2.bias']
    model.load_state_dict(current_state_dict)


def Load_Pretrained_MobileNetV2(model, model_type: Model_type) -> None:
    """
    Load MobileNet.
    Args:
        model: The model.
        model_type: The model layer_type.

    Returns: None

    """
    weights = model_type.weights().IMAGENET1K_V2
    new_check = weights.get_state_dict(progress=True)
    new_check['classifier.Head.0.weight'] = new_check['classifier.1.weight']
    del new_check['classifier.1.weight']
    new_check['classifier.Head.0.bias'] = new_check['classifier.1.bias']
    del new_check['classifier.1.bias']
    for name, param in model.state_dict().items():
        if name not in new_check.keys():
            new_check[name] = param

    model.load_state_dict(new_check)  # Loading the saved weights.


def Load_Pretrained_MobileNetV3(model, model_type: Model_type) -> None:
    """
    Load MobileNet.
    Args:
        model: The model.
        model_type: The model layer_type.

    Returns: None

    """
    weights = model_type.weights().IMAGENET1K_V2
    new_check = weights.get_state_dict(progress=True)
    new_check['classifier.Head.0.weight'] = new_check['classifier.3.weight']
    new_check['before_classifier.0.weight'] = new_check['classifier.0.weight']
    new_check['before_classifier.0.bias'] = new_check['classifier.0.bias']
    del new_check['classifier.3.weight']
    del new_check['classifier.0.bias']
    del new_check['classifier.0.weight']
    new_check['classifier.Head.0.bias'] = new_check['classifier.3.bias']
    del new_check['classifier.3.bias']
    for name, param in model.state_dict().items():
        if name not in new_check.keys():
            new_check[name] = param

    model.load_state_dict(new_check)  # Loading the saved weights.


def Load_Pretrained_model(model: nn.Module, model_type: Optional[Model_type]) -> None:
    """

    Args:
        model: The model.
        model_type: The model layer_type.

    Returns:

    """
    if model_type in [Model_type.MobileNetV2]:
        Load_Pretrained_MobileNetV2(model=model, model_type=model_type)
    if model_type is Model_type.MobileNetV3:
        Load_Pretrained_MobileNetV3(model, model_type)
    if model_type is Model_type.EfficientNet:
        Load_Pretrained_EfficientNet(model, model_type)
    if model_type in [Model_type.ResNet14, Model_type.ResNet18, Model_type.ResNet20, Model_type.ResNet32,
                      Model_type.ResNet34, Model_type.ResNet50, Model_type.ResNet101]:
        Load_pretrained_resnet(model=model)
