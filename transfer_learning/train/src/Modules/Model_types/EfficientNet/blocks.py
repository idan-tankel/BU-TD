"""
Efficient Model Block.
"""
from typing import Callable, List, Optional, Tuple
import argparse

import torch
from torch import nn, Tensor
from torchvision.ops import StochasticDepth
from torchvision.models.efficientnet import math, _MBConvConfig

from ...continual_learning_layers.module_blocks import Modulated_layer


class MBConvConfigOurs(_MBConvConfig):
    """
    Stores information listed at Table 1 of the EfficientNet paper & Table 4 of the EfficientNetV2 paper
    """

    def __init__(
            self,
            expand_ratio: float,
            kernel: int,
            stride: int,
            input_channels: int,
            out_channels: int,
            num_layers: int,
            width_mult: float = 1.0,
            depth_mult: float = 1.0,
            block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        input_channels = self.adjust_channels(input_channels, width_mult)
        out_channels = self.adjust_channels(out_channels, width_mult)
        num_layers = self.adjust_depth(num_layers, depth_mult)
        if block is None:
            block = MBConvOurs
        super().__init__(expand_ratio, kernel, stride, input_channels, out_channels, num_layers, block)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        """
        Adjust depth.
        Args:
            num_layers: The number of layers.
            depth_mult: The depth multiplier.

        Returns: The new depth.

        """
        return int(math.ceil(num_layers * depth_mult))


class Conv2dNormActivationOurs(nn.Module):
    """
    Conv block with masking and modulation.
    """

    def __init__(self, opts: argparse, masks: List, modulation: List, input_channels: int,
                 expanded_channels: int,
                 kernel_size: int,
                 norm_layer: nn.Module,
                 activation_layer: Optional[nn.Module],
                 stride: int = 1,
                 groups: int = 1,
                 dilation: int = 1
                 ):
        super(Conv2dNormActivationOurs, self).__init__()
        self.opts = opts
        self.ntasks = opts.data_set_obj['ntasks']
        padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=expanded_channels, kernel_size=kernel_size,
                              stride=stride, groups=groups, padding=padding, bias=False)
        self.mask_layer = Modulated_layer(opts=opts, layer=self.conv, masks=masks,
                                          task_embedding=modulation, create_modulation=True,
                                          create_masks=True, linear=False)
        self.bn = norm_layer(expanded_channels)
        self.relu = activation_layer

    def forward(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Compute forward.
        Args:
            inputs: The x, flag.

        Returns: The output, flag.

        """
        x, flags = inputs
        out = self.mask_layer(x, flags)
        out = self.bn(out)
        if self.relu is not None:
            out = self.relu()(out)
        return out, flags


class SqueezeExcitationOurs(torch.nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.

    Args:
        input_channels (int): Number of channels in the x image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., torch.nn.Module], optional): ``delta`` activation. Default: ``torch.nn.ReLU``
        scale_activation (Callable[..., torch.nn.Module]): ``sigma`` activation. Default: ``torch.nn.Sigmoid``
    """

    def __init__(
            self,
            opts,
            masking,
            modulation,
            input_channels: int,
            squeeze_channels: int,
            activation: Callable[..., torch.nn.Module] = torch.nn.ReLU,
            scale_activation: Callable[..., torch.nn.Module] = torch.nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Conv2d(in_channels=input_channels, out_channels=squeeze_channels, kernel_size=1)
        self.mask_layer1 = Modulated_layer(opts=opts, layer=self.fc1, masks=masking,
                                           task_embedding=modulation, create_masks=True,
                                           create_modulation=True, linear=False)
        self.fc2 = torch.nn.Conv2d(squeeze_channels, input_channels, 1)
        self.mask_layer2 = Modulated_layer(opts=opts, layer=self.fc2, masks=masking,
                                           task_embedding=modulation, create_masks=True,
                                           create_modulation=True, linear=False)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, x: Tensor, flags: Tensor) -> Tensor:
        """
        The scale block.
        Args:
            x: The input.
            flags: The flags.

        Returns:

        """
        scale = self.avgpool(x)
        scale = self.mask_layer1(scale, flags)
        scale = self.activation(scale)
        scale = self.mask_layer2(scale, flags)
        return self.scale_activation(scale)

    def forward(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.
        Args:
            inputs: The input, flags.

        Returns: The scale output, flags.

        """
        x, flags = inputs
        scale = self._scale(x, flags)
        return scale * x, flags


class MBConvOurs(nn.Module):
    """
    The basic block.
    """
    def __init__(
            self,
            opts,
            masking,
            modulation,
            cnf: MBConvConfigOurs,
            stochastic_depth_prob: float,
            norm_layer: Callable[..., nn.Module],
            se_layer: Callable[..., nn.Module] = SqueezeExcitationOurs,
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivationOurs(
                    opts=opts,
                    modulation=modulation,
                    masks=masking,
                    input_channels=cnf.input_channels,
                    expanded_channels=expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        layers.append(
            Conv2dNormActivationOurs(
                opts=opts,
                modulation=modulation,
                masks=masking,
                input_channels=expanded_channels,
                expanded_channels=expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        # self.ext_layer =
        layers.append(se_layer(opts=opts,
                               modulation=modulation,
                               masking=masking, input_channels=expanded_channels, squeeze_channels=squeeze_channels,
                               activation=nn.SiLU))

        # project
        layers.append(
            Conv2dNormActivationOurs(
                opts=opts, modulation=modulation, masks=masking, input_channels=expanded_channels, expanded_channels=
                cnf.out_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Forward pass.
        Args:
            inputs: The input, flags.

        Returns: The scale output, flags.

        """
        x, flags = inputs
        result = self.block((x, flags))
        if self.use_res_connect:
            result, _ = result
            result = self.stochastic_depth(result)
            result += x
            result = result, flags
        return result
