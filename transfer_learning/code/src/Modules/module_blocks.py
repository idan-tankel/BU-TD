"""
Module blocks.
"""
import torch.nn as nn

from typing import Callable, Type, Optional, List
import argparse

from torch import Tensor

import torch

from torchvision.models.resnet import conv1x1

import torch.nn.functional as F

from src.Utils import Expand
from src.Modules.Batch_norm import BatchNorm


class DownSample(nn.Module):
    """
    Downsample the input for residual connection.
    """

    def __init__(self, norm_layer: Type[BatchNorm], inplanes: int, planes: int, expansion: int, stride: int,
                 ntasks: int):
        """
        Downsample layer.
        Args:
            norm_layer: The norm layer.
            inplanes: Inplanes.
            planes: Outplanes.
            expansion: The block expansion.
            stride: The stride.
            ntasks: The number of tasks.
        """
        super(DownSample, self).__init__()
        self.conv1x1 = conv1x1(inplanes, planes * expansion, stride)
        self.norm = norm_layer(planes * expansion, ntasks)

    def forward(self, x: Tensor, flags: Tensor) -> Tensor:
        """

        Args:
           x: The input.
           flags: The task flag.

        Returns: The down-sampled input.

        """
        out = self.conv1x1(x)
        out = self.norm(out, flags)
        return out


class Modulate_weight_keep_grads(nn.Module):
    """
    Modulate current weight in a differentiable manner
    to keep gradients.
    """

    def __init__(self, weight: Tensor, additive: bool):
        """
        Modulates current pretrained weight.
        Args:
            weight: The pretrained weight.
            additive: Whether the modulations are additive or multiplicative.
        """
        super(Modulate_weight_keep_grads, self).__init__()
        self.weight = weight
        self.additive = additive

    def forward(self, emb: Tensor) -> Tensor:
        """
        Modulates the current weight with the input.
        Args:
            emb: The embedding weight.

        Returns: The embedded weight.

        """
        if self.additive:
            return self.weight + emb
        else:
            return self.weight * (1 - emb)


class WeightModulation(nn.Module):
    """
    NeuralModulation layer.
    Create Channel or Column modulations layer.
    The main idea of the paper allowing continual learning without forgetting.
    """

    def __init__(self, opts: argparse, layer: nn.Conv2d, modulations: Optional[List]):
        """

        Args:
            opts: The model options.
            layer: The current later to modulate its weight.
        """
        super(WeightModulation, self).__init__()
        self.ntasks = opts.data_set_obj.ntasks
        self.layer = layer
        self.modulation_factor = opts.data_set_obj.weight_modulation_factor
        self.weight = Modulate_weight_keep_grads(weight=self.layer.weight, additive=False)
        (c_in, c_out, k1, k2) = layer.weight.shape
        (mod1, mod2, mod3, mod4) = self.modulation_factor
        size = (c_in // mod1, c_out // mod2, k1 // mod3, k2 // mod4)
        self.modulation = nn.ParameterList([nn.Parameter(torch.Tensor(*size), requires_grad=True) for _ in range(
            self.ntasks)])
        for index in range(self.ntasks):
            modulations[index].append(self.modulation[index])

    def forward(self, x, flags: Tensor) -> Tensor:
        """
        perform the channel/pixel modulations.
        Args:
            x: Torch of shape [B,C,H,W] to modulate.
            flags: The task id.

        Returns: Torch of shape [B,C,H,W], the modulated tensor.

        """
        task_id = flags[0].argmax().item()
        task_emb = self.modulation[task_id]  # compute the task embedding according to the direction_idx.
        task_emb = Expand(task_emb, shapes=self.modulation_factor)
        weight = self.weight(task_emb)

        output = F.conv2d(input=x, weight=weight, stride=self.layer.stride,
                          padding=self.layer.padding, dilation=self.layer.dilation, groups=self.layer.groups)
        return output


class LambdaLayer(nn.Module):
    """
    Lambda layer to support non-basic downsample strategy.
    """

    def __init__(self, lambda_fun: Callable):
        super(LambdaLayer, self).__init__()
        self.lamda = lambda_fun

    def forward(self, x: Tensor, flags: Tensor) -> Tensor:
        """

        Args:
            x: The input.
            flags: The task flag, not really used.

        Returns:

        """
        return self.lamda(x)
