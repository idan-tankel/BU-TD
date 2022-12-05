import argparse

import torch
import torch.nn as nn

from training.Utils import flag_to_idx


# Here we define all module building blocks heritages from nn.Module.

class Depthwise_separable_conv(nn.Module):
    # More saving version of Conv.
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
        self.depthwise = nn.Conv2d(channels_in, channels_in, kernel_size=kernel_size, stride=stride, padding=padding,
                                   groups=channels_in,
                                   bias=bias)  # Preserves the number of channels but may downsample by stride.
        self.pointwise = nn.Conv2d(channels_in, channels_out, kernel_size=(1, 1),
                                   bias=bias)  # Preserves the inner channels but may change the number of channels.

    def forward(self, x: torch) -> torch:
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

    return Depthwise_separable_conv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)


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
    layer = conv3x3(in_channels, out_channels)  # Changing the number of channels.
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
    return nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride, bias=bias)


class Modulation_and_Lat(nn.Module):
    # performs the lateral connection BU1 -> TD or TD -> BU2.
    # Applies channel modulation, BN, ReLU on the lateral connection.
    # Then perform the lateral connection to the input and then more relu is applied.

    def __init__(self, opts: argparse, filters: int):
        """
        Args:
            opts: The model options.
            filters: The number of filters.
        """
        super(Modulation_and_Lat, self).__init__()
        shape = [filters, 1, 1]
        self.side = nn.Parameter(
            torch.Tensor(*shape))  # creates the learnable parameter of shape [nchannels,1,1] according to nchannels.
        self.norm_and_relu = nn.Sequential(opts.norm_layer(opts, filters), opts.activation_fun())
      #  self.norm = opts.norm_layer(opts, filters)  # batch norm after the channel-modulation of the lateral.
     #   self.relu1 = opts.activation_fun()  # activation_fun after the batch_norm layer
        self.relu = opts.activation_fun()  # activation_fun after the skip connection

    def forward(self, x:torch, lateral:torch) -> torch:
        """
        Args:
            x: The model input.
            lateral: The previous stream lateral connection, of the same shape.

        Returns: The output after the lateral connection.

        """
        side_val = lateral * self.side  # channel-modulation(CM)
      #  side_val = self.norm(side_val)  # Batch_norm after the CM
       # side_val = self.relu1(side_val)  # Activation_fun after the batch_norm
        side_val = self.norm_and_relu(side_val)
        x = x + side_val  # The lateral skip connection
        x = self.relu(x)  # Activation_fun after the skip connection
        return x

class Modulation(nn.Module):  # Modulation layer.
    def __init__(self, opts: argparse, shape: list, pixel_modulation: bool, task_embedding: list):
        """
        Channel & pixel modulation layer.
        The main idea of the paper allowing continual learning without forgetting.
        Args:
            opts: The model options.
            shape: The shape to create the model according to.
            pixel_modulation: Whether to create pixel/channel modulation.
        """
        super(Modulation, self).__init__()
        self.opts = opts # Store the model opts.
        self.modulations = nn.ParameterList()  # Module list containing modulation for all directions.
        if pixel_modulation:
            size = [1, *shape]  # If pixel modulation matches the inner spatial of the input
        else:
            size = [shape, 1, 1]  # If channel modulation matches the number of channels
        for i in range(opts.ndirections):  # allocating for every direction its task embedding
            layer = nn.Parameter(torch.Tensor(*size)) # The task embedding.
            task_embedding[i].append(layer) # Add to the learnable parameters.
            self.modulations.append(layer) # Add to the modulation list.

    def forward(self, x: torch, flag: torch) -> torch:
        """
        perform the channel/pixel modulation.
        Args:
            x: Torch of shape [B,C,H,W] to modulate.
            flag: torch of shape [B,S].

        Returns: Torch of shape [B,C,H,W], the modulated tensor.

        """
        direction_id = flag_to_idx(flag)  # Compute the index of the one-hot.
        task_emb = self.modulations[direction_id]  # compute the task embedding according to the direction_idx.
        output = x * (1 - task_emb)  # perform the modulation.
        return output
