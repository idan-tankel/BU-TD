"""
Here we define all module building blocks heritages from nn.Module.
"""
import argparse

import torch
import torch.nn as nn
from torch import Tensor

from training.Utils import flag_to_idx


class Depthwise_separable_conv(nn.Module):
    """
    More efficient version of Conv.
    """

    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, stride: int = 1, padding: int = 1,
                 bias: bool = False):
        """
        Args:
            channels_in: In channels of the input tensor.
            channels_out: Out channels of the input tensor.
            kernel_size: The kernel size.
            stride: The stride.
            padding: The padding.
            bias: Whether to use bias.
        """
        super(Depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels=channels_in, out_channels=channels_in, kernel_size=kernel_size,
                                   stride=stride, padding=padding,
                                   groups=channels_in,
                                   bias=bias)  # Preserves the number of channels but may downsample by stride.
        self.pointwise = nn.Conv2d(in_channels=channels_in, out_channels=channels_out, kernel_size=(1, 1),
                                   bias=bias)  # Preserves the inner channels but may change the number of channels.

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor to the Conv.

        Returns: Output tensor from the conv.

        """
        out = self.depthwise(x)  # Downsample the tensor.
        out = self.pointwise(out)  # Change the number of channels if needed.
        return out


def conv3x3(in_channels: int, out_channels: int, stride: int = 1, bias=False) -> nn.Module:
    """
    Create specific version of Depthwise_separable_conv with kernel equal 3.
    Args:
        in_channels: In channels of the input tensor
        out_channels: Out channels of the output tensor.
        stride: The stride.
        bias: Whether to use the bias.

    Returns: Module that performs the conv3x3.

    """

    return Depthwise_separable_conv(channels_in=in_channels, channels_out=out_channels, kernel_size=3, stride=stride,
                                    bias=bias)


def conv3x3up(in_channels: int, out_channels: int, size: tuple, upsample=False) -> nn.Module:
    """
    Upsampling version of Conv3x3.
    Args:
        in_channels: The number of channels in the input. out_channels < in_channels
        out_channels: The number of channels in the output. out_channels < in_channels
        size: The size to upsample to.
        upsample: Whether to upsample.

    Returns: Module that upsample the tensor.

    """
    layer = conv3x3(in_channels=in_channels, out_channels=out_channels)  # Changing the number of channels.
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
            opts: The model_test options.
            nfilters: The number of filters.
        """
        super(Modulation_and_Lat, self).__init__()
        shape = [nfilters, 1, 1]
        self.side = nn.Parameter(
            torch.Tensor(*shape))  # creates the learnable parameter of shape [nchannels, 1, 1] according to nchannels.
        self.norm_and_relu = nn.Sequential(opts.norm_layer(opts, nfilters), opts.activation_fun())
        self.relu = opts.activation_fun()  # activation_fun after the skip connection

    def forward(self, x: Tensor, flags: Tensor, lateral: Tensor) -> Tensor:
        """
        Args:
            x: The model_test input.
            flags: The flags, needed for BN.
            lateral: The previous stream lateral connection, of the same shape.

        Returns: The output after the lateral connection.

        """
        side_val = lateral * self.side  # channel-modulation(CM)
        side_val = self.norm_and_relu[0](inputs=side_val, flags=flags)
        side_val = self.norm_and_relu[1](input=side_val)
        x = x + side_val  # The lateral skip connection
        x = self.relu(x)  # Activation_fun after the skip connection
        return x


class Modulation(nn.Module):
    """
    Modulation layer.
    Create Channel or Column modulation layer.
    The main idea of the paper allowing continual learning without forgetting.
    """

    def __init__(self, opts: argparse, shape: list, column_modulation: bool, task_embedding: list):
        """

        Args:
            opts: The model_test options.
            shape: The shape to create the model_test according to.
            column_modulation: Whether to create pixel/channel modulation.
        """
        super(Modulation, self).__init__()
        self.opts = opts  # Store the model_test model_opts.
        self.modulations = nn.ParameterList()  # Module list containing modulation for all directions.
        if column_modulation:
            size = [1, *shape]  # If pixel modulation matches the inner spatial of the input
        else:
            size = [shape, 1, 1]  # If channel modulation matches the number of channels
        for i in range(opts.ndirections):  # allocating for every list_task_structs its list_task_structs embedding
            layer = nn.Parameter(torch.Tensor(*size))  # The list_task_structs embedding.
            task_embedding[i].append(layer)  # Add to the learnable parameters.
            self.modulations.append(layer)  # Add to the modulation list.

    def forward(self, x: Tensor, flags: Tensor) -> Tensor:
        """
        perform the channel/pixel modulation.
        Args:
            x: Torch of shape [B,C,H,W] to modulate.
            flags: torch of shape [B,S].

        Returns: Torch of shape [B,C,H,W], the modulated tensor.

        """
        direction_id = flag_to_idx(flags=flags)  # Compute the index of the one-hot.
        #    print(direction_id)
        task_emb = self.modulations[direction_id]  # compute the list_task_structs embedding according to the direction_idx.
        output = x * (1 - task_emb)  # perform the modulation.
        return output
