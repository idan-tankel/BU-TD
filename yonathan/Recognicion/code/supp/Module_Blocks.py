import torch
import torch.nn as nn
import argparse
import numpy as np
from supp.utils import flag_to_idx

# Here we define all module building blocks heritages from nn.Module.

class Depthwise_separable_conv(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, stride: int = 1, padding: int = 1,
                 bias: bool = False):
        """
        Create a parameters preserving version of conv.
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
        self.pointwise = nn.Conv2d(channels_in, channels_out, kernel_size=1,
                                   bias=bias)  # Preserves the inner channels but changes the number of channels.

    def forward(self, x: torch) -> torch:
        """
        Args:
            x: Input tensor to the Conv.

        Returns: Output tensor from the conv.

        """
        out = self.depthwise(x)  # Downsample the tensor.
        out = self.pointwise(out)  # Change the number of channels if needed.
        return out


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Module:
    """
    Create specific version of Depthwise_separable_conv with kernel = 3.
    Args:
        in_channels: In channels of the input tensor
        out_channels: Out channels of the output tensor.
        stride: The stride.

    Returns: Module that performs the conv3x3.

    """

    return Depthwise_separable_conv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


def conv3x3up(in_channels: int, out_channels: int, size: tuple, upsample=False) -> nn.Module:
    """
    Upsampling version of Conv3x3.
    Args:
        in_channels: The number of channels in the input. out_channels < in_channels
        out_channels: The number of channels in the output. out_channels < in_channels
        size: The size to upsample to.
        upsample: Whether to upsample.

    Returns: Module that Upsamples the tensor.

    """
    layer = conv3x3(in_channels, out_channels)  # Changing the number of channels.
    if upsample:  # Adding upsampling layer.
        layer = nn.Sequential(nn.Upsample(size=size, mode='bilinear', align_corners=False),
                              layer)  # Upsample the inner dimensions of the tensor.
    return layer


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Module:
    """
    return 1x1 conv.
    Args:
        in_channels: In channels of the input tensor
        out_channels: Out channels of the output tensor.
        stride: The stride.

    Returns: 1 by 1 conv.

    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class ChannelModulationForLateralConnections(nn.Module):
    # The layer performs the channel modulation on the lateral connection.
    def __init__(self, nchannels: int):
        """
        Args:
            nchannels: The number of channels to perform channel-modulation on.
        """
        super(ChannelModulationForLateralConnections, self).__init__()
        self.nchannels = nchannels
        shape = [self.nchannels, 1, 1]
        self.weights = nn.Parameter(
            torch.Tensor(*shape))  # creates the learnable parameter of shape [nchannels,1,1] according to nchannels.

    def forward(self, inputs: torch) -> torch:
        """
        Perform the channel modulation.
        Args:
            inputs: Tensor of shape [N,C,H,W].

        Returns: Tensor of shape [N,C,H,W].

        """
        return inputs * self.weights  # performs the channel wise modulation.


class SideAndComb(nn.Module):
    # performs the lateral connection BU1 -> TD or TD -> BU2.
    # Applies channel modulation, BN, relu on the lateral connection.
    # Then perform the lateral connection to the input and then more relu is applied.

    def __init__(self, opts: argparse, filters: int):
        """
        Args:
            opts: The model options.
            filters: The number of filters.
        """
        super(SideAndComb, self).__init__()
        self.side = ChannelModulationForLateralConnections(filters)  # channel-modulation layer.
        self.norm = opts.norm_layer(opts, filters)  # batch norm after the channel-modulation of the lateral.
        self.relu1 = opts.activation_fun()  # activation_fun after the batch_norm layer
        self.relu2 = opts.activation_fun()  # activation_fun after the skip connection

    def forward(self, inputs: torch) -> torch:
        """
        Args:
            inputs: Two tensors, the first is the input and the second is the lateral connection.

        Returns: The output after the lateral connection.

        """
        x, lateral = inputs  # input, lateral connection.
        side_val = self.side(lateral)  # channel-modulation(CM)
        side_val = self.norm(side_val)  # batch_norm after the CM
        side_val = self.relu1(side_val)  # activation_fun after the batch_norm
        x = x + side_val  # the lateral skip connection
        x = self.relu2(x)  # activation_fun after the skip connection
        return x

class Modulation(nn.Module):  # Modulation layer.
    def __init__(self, opts, shape: list, pixel_modulation: bool, task_embedding: list):
        """
        Channel & pixel modulation layer.
        The main idea of the paper allowing continual learning without forgetting.
        Args:
            opts: The model options.
            shape: The shape to create the model according to.
            pixel_modulation: Whether to create pixel/channel modulation.
        """
        super(Modulation, self).__init__()
        self.modulations = nn.ParameterList()  # Module list containing modulation for all directions.
        if pixel_modulation:
            self.size = [-1, 1, *shape]  # If pixel modulation matches the inner spatial of the input
        else:
            self.size = [-1, shape, 1, 1]  # If channel modulation matches the number of channels
        inshapes = np.prod(shape)  # Compute the shape needed for initializing.
        for i in range(opts.ndirections):  # allocating for every task its task embedding
            weight = nn.Parameter(torch.Tensor(1, inshapes))
            task_embedding[i].extend(weight)  # Save the task embedding.
            self.modulations.append(weight)

    def forward(self, inputs: torch, flag: torch) -> torch:
        """
        perform the channel/pixel modulation.
        Args:
            inputs: Torch of shape [B,C,H,W] to modulate.
            flag: torch of shape [B,S].

        Returns: Torch of shape [B,C,H,W], the modulated tensor.

        """
        direction_id = flag_to_idx(flag)  # Compute the index of the one-hot.
        task_emb = self.modulations[direction_id].view(self.size)  # compute the task embedding according to the task_idx.
        inputs = inputs * (1 - task_emb)  # perform the modulation.
        return inputs

def init_module_weights(modules: list[nn.Module]) -> None:
    """
    Initializing the module weights according to the original BU-TD paper.
    Args:
        modules: All model's layers

    """
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, ChannelModulationForLateralConnections):
            nn.init.xavier_uniform_(m.weights)

        elif isinstance(m, Modulation):
            for layer in m.modulations:
             nn.init.xavier_uniform_(layer)