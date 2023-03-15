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

from torch.nn.parameter import Parameter


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
        (c_in, c_out, k1, k2) = layer.weight.shape
        (mod1, mod2, mod3, mod4) = self.modulation_factor
        size = (c_in // mod1, c_out // mod2, k1 // mod3, k2 // mod4)
        self.modulation = nn.ParameterList([nn.Parameter(torch.Tensor(*size), requires_grad=True) for _ in range(
            self.ntasks)])
        for index in range(self.ntasks):
            modulations[index].append(self.modulation[index])

    def forward(self, flags: Tensor) -> Tensor:
        """
        Perform the weights modulations.
        Args:
            flags: The task id.

        Returns: Torch of shape [B,C,H,W], the modulated tensor.

        """
        task_id = flags[0].argmax().item()
        task_emb = self.modulation[task_id]  # compute the task embedding according to the direction_idx.
        task_emb = Expand(task_emb, shapes=self.modulation_factor)
        weight = self.layer.weight * (1 - task_emb)
        return weight


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


# DEFAULT_THRESHOLD = 5e-3

class Binary_masking(torch.autograd.Function):
    """
    Binary masking object.
    """

    @staticmethod
    def forward(ctx, inputs, DEFAULT_THRESHOLD):
        """

        Args:
            ctx: The ctx.
            inputs: The inputs.
            DEFAULT_THRESHOLD: The threshold.

        Returns:

        """
        outputs = inputs.clone()
        outputs[inputs.le(DEFAULT_THRESHOLD)] = 0
        outputs[inputs.gt(DEFAULT_THRESHOLD)] = 1
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs: list):
        """

        Args:
            ctx: The crx.
            *grad_outputs: The gradient output.

        Returns:

        """
        return grad_outputs[0], None


class MaskWeight(nn.Module):
    """
    Mask weight.
    """

    def __init__(self, opts: argparse, layer: nn.Conv2d, mask: list):
        super(MaskWeight, self).__init__()
        self.weight = layer.weight
        self.mask_scale = 1e-2
        self.stride = layer.stride
        self.padding = layer.padding
        self.dilation = layer.dilation
        self.groups = layer.groups
        self.opts = opts
        self.ntasks = self.opts.data_set_obj.ntasks
        self.mask_real = self.weight.data.new(self.weight.size())
        self.mask_real.fill_(self.mask_scale)
        # mask_real is now a trainable parameter.
        self.mask_real = Parameter(self.mask_real)
        self.masks = nn.ParameterList()
        for i in range(self.ntasks):
            layer = Parameter(self.weight.data.new(self.weight.size()))
            mask[i].append(layer)
            self.masks.append(layer)

    def forward(self, flags):
        """
        Forward the mask.
        Args:
            flags: The input.

        Returns: The masked conv.

        """
        task_id = flags[0]
        Mask = Binary_masking.apply(self.mask_real[task_id], self.opts.data_set_obj.threshold)
        new_weight = Mask * self.weight
        return new_weight


class conv_with_modulation_and_masking(nn.Module):
    """
    Conv with modulation and masking.
    """

    def __init__(self, opts: argparse, conv_layer: nn.Module, task_embedding: list, create_modulation: bool,
                 create_masks: bool, masks: list):
        """

        Args:
            opts: The model opts.
            conv_layer: The conv layer.
            task_embedding: The task embedding.
            create_modulation: Whether to create modulation.
            create_masks: Whether to create masks.
        """
        super(conv_with_modulation_and_masking, self).__init__()
        self.opts = opts
        self.create_modulation = create_modulation
        self.create_masks = create_masks
        self.ntasks = opts.data_set_obj.ntasks
        self.modulate_weights = opts.weight_modulation
        self.mask_weights = opts.mask_weights
        self.layer: nn.Conv2d = conv_layer
        if create_modulation:
            self.modulation = WeightModulation(opts=opts, layer=conv_layer, modulations=task_embedding)

        if create_masks:
            self.masks = MaskWeight(opts=opts, layer=conv_layer, mask=masks)

    def forward(self, x: Tensor, flags: Tensor) -> Tensor:
        """
        perform the channel/pixel modulation.
        Args:
            x: Torch of shape [B,C,H,W] to modulate.
            flags: The samples.

        Returns: Torch of shape [B,C,H,W], the modulated tensor.

        """
        weights = self.layer.weight
        if self.modulate_weights and self.create_modulation:
            weights = self.modulation(flags=flags)
        if self.create_masks and self.mask_weights:
            weights = self.masks(flags=flags)
        output = F.conv2d(input=x, weight=weights, stride=self.layer.stride,
                          padding=self.layer.padding, dilation=self.layer.dilation, groups=self.layer.groups)
        return output
