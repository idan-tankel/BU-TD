import torch
import torch.nn as nn

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
