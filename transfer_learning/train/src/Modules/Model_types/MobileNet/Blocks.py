"""
Conv, norm, activation block.
"""
import torch
from typing import Union, Tuple, Optional, Callable, Sequence
import warnings
from torchvision.ops.misc import _make_ntuple
import argparse

import torch.nn as nn


class ConvNormActivation(nn.Module):
    """
    Create Conv, Norm, Activation layer.
    """

    def __init__(
            self,
            opts: argparse,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, ...]] = 3,
            stride: Union[int, Tuple[int, ...]] = 1,
            padding: Optional[Union[int, Tuple[int, ...], str]] = None,
            groups: int = 1,
            norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
            activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
            dilation: Union[int, Tuple[int, ...]] = 1,
            inplace: Optional[bool] = True,
            bias: Optional[bool] = None,
            conv_layer: Callable[..., torch.nn.Module] = torch.nn.Conv2d,
    ) -> None:
        super(ConvNormActivation, self).__init__()

        if padding is None:
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                _conv_dim = len(kernel_size) if isinstance(kernel_size, Sequence) else len(dilation)
                kernel_size = _make_ntuple(kernel_size, _conv_dim)
                dilation = _make_ntuple(dilation, _conv_dim)
                padding = tuple((kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim))
        if bias is None:
            bias = norm_layer is None

        self.conv = conv_layer(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        if norm_layer is not None:
            self.norm_layer = norm_layer(out_channels, opts.ntasks)

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            self.activation = activation_layer(**params)

        self.out_channels = out_channels


