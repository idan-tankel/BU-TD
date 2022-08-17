import os
import torch
import torch.nn as nn
from .FlagAt import *
import argparse


def folder_size(path: str) -> int:
    """

    :param path:path to a language file.
    :return: number of characters in the language
    """
    return len([_ for _ in os.scandir(path)])


def create_dict(path: str) -> dict:
    """

    :param path: path to all raw Omniglot languages
    :return: dictionary of number of characters per language
    """
    dict_language = {}
    cnt = 0
    for ele in os.scandir(path):  # for language in Omniglot_raw find the number of characters in it.
        dict_language[cnt] = folder_size(ele)  # Find number of characters in the folder.
        cnt += 1
    return dict_language


def instruct(struct: argparse, key: str) -> bool:
    """
    :param struct: a parser
    :param key: key to check in the parser
    :return: True iff key is one of the struct's keys.
    """
    return getattr(struct, key, None) is not None


def flag_to_task(flag: torch) -> int:
    """
    :param flag: The flag.
    :return: The task the model solves.
    """
    task = torch.argmax(flag, dim=1)[0]  # Finds the non zero entry in the one-hot vector
    return task


def get_laterals(laterals: list[torch], layer_id: int, block_id: int) -> torch:
    """
    :param laterals: All lateral connections from the previous stream, if exists.
    :param layer_id: The layer id in the stream.
    :param block_id: The block id in the layer.
    :return: all the lateral connections associate with the block(usually 3).
    """
    if laterals is None:  # If BU1, there are not any lateral connections.
        return None
    if len(laterals) > layer_id:  # assert we access to an existing layer.
        layer_laterals = laterals[layer_id]  # Get all lateral associate with the layer.
        if type(layer_laterals) == list and len(
                layer_laterals) > block_id:  # If there are some several blocks in the layer we access according to block_id.
            return layer_laterals[block_id]  # We return all lateral associate with the block_id.
        else:
            return layer_laterals  # If there is only 1 lateral connection in the block we return it.
    else:
        return None


class depthwise_separable_conv(nn.Module):
    """
    Layer downsampling the tensor .
    Equivalent to conv3x3(channels_in,channels_out) but with much less parameters.
    """

    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, stride: int = 1, padding: int = 1,
                 bias: bool = False) -> None:
        """
        :param channels_in: Channels in of the input tensor.
        :param channels_out: Channels out of the output tensor
        :param kernel_size: Kernel size
        :param stride: stride to Downsample the tensor
        :param padding: Padding
        :param bias: Whether to use bias or not
        """
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(channels_in, channels_in, kernel_size=kernel_size, stride=stride, padding=padding,
                                   groups=channels_in,
                                   bias=False)  # Preserves the number of channels but may downsample by stride.
        self.pointwise = nn.Conv2d(channels_in, channels_out, kernel_size=1,
                                   bias=bias)  # Preserves the inner channels but changes the number of channels.

    def forward(self, x: torch) -> torch:
        """
        :param x: tensor of shape [B,C_in,H1,W1]
        :return: [B,C_out,H2,W2]
        """
        out = self.depthwise(x)  # Downsample the tensor.
        out = self.pointwise(out)  # Change the number of channels
        return out


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Module:
    """
    :param in_channels: In channels of the input tensor
    :param out_channels: Out channels of the output tensor.
    :param stride: stride.
    :return: Module that performs the 3x3conv.
    """
    return depthwise_separable_conv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


def conv3x3up(in_channels, out_channels,size, upsample_size=1) -> nn.Module:
    """
    Opposite to the conv3x3: it decreases the number of channels, and increases the inner channels.
    :param in_channels: The number of channels in the input. out_channels < in_channels
    :param out_channels: The number of channels in the output. out_channels < in_channels
    :param upsample_size: The factor to upsample the inner dimensions.
    :return: Module that Upsamples the tensor.
    """
    layer = conv3x3(in_channels, out_channels)  # Changing the number of channels.
    if upsample_size > 1:
        layer = nn.Sequential(nn.Upsample(size = size, mode='bilinear', align_corners=False),
                              layer)  # Upsample the inner dimensions of the tensor.
    return layer


def conv1x1(in_channels, out_channels, stride=1) -> nn.Module:
    """
    :param in_channels: In channels of the input tensor
    :param out_channels: Out channels of the output tensor.
    :param stride: stride.
    :return: Module: the layer just changes the number of channels, the inner dimensions remain the same.
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


def num_params(params: list) -> int:
    """
    :params params: List of all model's parameters.
    :return: Number of leaned parameters.
    """
    nparams = 0
    for param in params:  # For each parameter in the model we sum its parameters
        cnt = 1
        for p in param.shape:  # The number of params in each weight is the product if its shape.
            cnt = cnt * p
        nparams = nparams + cnt  # Sum for all params.
    return nparams
