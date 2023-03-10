"""
Here we define all module building blocks heritages from nn.Module.
"""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from ..Data.Structs import inputs_to_struct
from ..Utils import Expand, tuple_direction_to_index
from .Batch_norm import BatchNorm

def conv3x3(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, bias=False,
            padding=1) -> nn.Module:
    """
    Create specific version of Depthwise_separable_conv with kernel equal 3.
    Args:
        in_channels: In channels of the input tensor
        out_channels: Out channels of the output tensor.
        kernel_size: The kernel size.
        stride: The stride.
        bias: Whether to use the bias.
        padding: The padding


    Returns: Module that performs the conv3x3.

    """
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     bias=bias, padding=padding)


def conv3x3up(in_channels: int, out_channels: int, size: tuple, kernel_size: int, upsample=False,
              padding=1) -> nn.Module:
    """
    Upsampling version of Conv3x3.
    Args:
        in_channels: The number of channels in the input. out_channels < in_channels
        out_channels: The number of channels in the output. out_channels < in_channels
        kernel_size: The kernel size.
        size: The size to upsample to.
        upsample: Whether to upsample.
        padding: The padding.

    Returns: Module that upsample the tensor.

    """
    layer = conv3x3(in_channels=in_channels, out_channels=out_channels,
                    padding=padding, kernel_size=kernel_size)  #
    # Changing the number of
    # channels.
    if upsample:  # Adding upsample layer.
        layer = nn.Sequential(nn.Upsample(size=size, mode='bilinear', align_corners=False),
                              layer)  # Upsample the inner dimensions of the tensor.
    return layer


def conv1x1(in_channels: int, out_channels: int, stride: int = 1, bias=False) -> nn.Module:
    """
    return 1x1 conv.
    Args:
        in_channels: In channels of the input tensor
        out_channels: Out channels of the output tensor.
        stride: The stride.
        bias: Whether to use bias.

    Returns: 1 by 1 conv.

    """
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=stride, bias=bias)


class Modulation_and_Lat(nn.Module):
    """
    performs the lateral connection BU1 -> TD or TD -> BU2.
    Applies channel modulation, BN, ReLU on the lateral connection.
    Then perform the lateral connection to the input and then more relu is applied.
    """

    def __init__(self, opts: argparse, nfilters: int):
        """
        Args:
            opts: The model options.
            nfilters: The number of filters.
        """
        super(Modulation_and_Lat, self).__init__()
        shape = [nfilters, 1, 1]
        self.relu = nn.ReLU(inplace=True)
        self.side = nn.Parameter(
            torch.Tensor(*shape))  # creates the learnable parameter of shape [nchannels, 1, 1] according to nchannels.
        self.norm1 = BatchNorm(opts, nfilters)

    def forward(self, x: Tensor, samples: Tensor, lateral: Tensor) -> Tensor:
        """
        Args:
            x: The model input.
            flags: The samples, needed for BN.
            lateral: The previous stream lateral connection, of the same shape.

        Returns: The output after the lateral connection.

        """
        side_val = lateral * self.side  # channel-modulation(CM)
        side_val = self.norm1(inputs=side_val, samples = samples)
        side_val = self.relu(input=side_val)
        x = x + side_val  # The lateral skip connection
        x = self.relu(x)  # Activation_fun after the skip connection
        return x

class conv_with_modulation(nn.Module):
    """
    Create specific version of Depthwise_separable_conv with kernel equal 3.
    Args:
        in_channels: In channels of the input tensor
        out_channels: Out channels of the output tensor.
        kernel_size: The kernel size.
        stride: The stride.
        bias: Whether to use the bias.
        padding: The padding


    Returns: Module that performs the conv3x3.

    """
    def __init__(self,opts:argparse,conv_layer:nn.Module,task_embedding:list,create_modulation:bool):
        super(conv_with_modulation, self).__init__()
        self.opts = opts
        self.create_modulation = create_modulation
        self.ntasks = opts.data_obj.ndirections
        self.modulate_weights = opts.weight_modulation
        self.layer: nn.Conv2d = conv_layer
        (c_in, c_out, k1, k2) = self.layer.weight.shape
        (mod1, mod2, mod3, mod4) = opts.weight_modulation_factor
        self.modulation_factor = opts.weight_modulation_factor
        size = (c_in // mod1, c_out // mod2, k1 // mod3, k2 // mod4)
        if create_modulation:
            self.modulation = nn.ParameterList()
            for i in range(self.ntasks):
                layer = nn.Parameter(torch.Tensor(*size), requires_grad=True)
                self.modulation.append(layer)
                task_embedding[i].append(layer)

    def forward(self, x: Tensor, samples: inputs_to_struct) -> Tensor:
        """
        perform the channel/pixel modulation.
        Args:
            x: Torch of shape [B,C,H,W] to modulate.
            samples: The samples.

        Returns: Torch of shape [B,C,H,W], the modulated tensor.

        """
        weights = self.layer.weight
        if self.modulate_weights and self.create_modulation:
            task = samples.direction_idx[0]
            task_emb = self.modulation[task]  # compute the task embedding according to the direction_idx.
            task_emb = Expand(task_emb, shapes=self.modulation_factor)
            weights = task_emb * weights
        output = F.conv2d(input=x, weight=weights, stride=self.layer.stride,
                          padding=self.layer.padding, dilation=self.layer.dilation, groups=self.layer.groups)
        return output

