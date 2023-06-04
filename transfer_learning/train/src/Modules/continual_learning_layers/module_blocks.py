"""
Module blocks.
"""
import torch.nn as nn

from typing import Optional, List
import argparse

from torch import Tensor

import torch

import torch.nn.functional as F

from ...Utils import Expand, compute_size

from torch.nn.parameter import Parameter


class WeightModulation(nn.Module):
    """
    NeuralModulation layer.
    Create Channel or Column modulations layer.
    The main idea of the paper allowing continual learning without forgetting.
    """

    def __init__(self, opts: argparse, layer: nn.Module, modulations: Optional[List], linear: bool, shape: list =
    None):
        """

        Args:
            opts: The Model options.
            layer: The current later to modulate its weight.
        """
        super(WeightModulation, self).__init__()
        self.opts = opts
        self.ntasks = opts.ntasks
        self.layer = layer
        self.modulation_factor = opts.data_set_obj['weight_modulation_factor'] if shape is None else shape
        self.modulation = nn.ParameterList()
        if linear:
            (c_in, c_out) = layer.weight.shape
            (mod1, mod2) = self.modulation_factor
            size = (c_in // mod1, c_out // mod2)
            self.modulation = nn.ParameterList([nn.Parameter(torch.Tensor(*size), requires_grad=True) for _ in range(
                self.ntasks)])
        else:
            size = compute_size(layer.weight.shape, self.modulation_factor)
            self.modulation = nn.ParameterList([nn.Parameter(torch.Tensor(*size), requires_grad=True) for _ in range(
                self.ntasks)])
            if modulations is not None:
                for i in range(self.ntasks):
                    modulations[i].append(self.modulation[i])

    def forward(self, flags: Tensor) -> Tensor:
        """
        Perform the weights modulations.
        Args:
            flags: The task id.

        Returns: Torch of shape [B,C,H,W], the modulated tensor.

        """
        weight = self.layer.weight
        task_id = flags[0]
        task_emb = self.modulation[task_id]
        if len(task_emb) == 4:
            task_emb = Expand(opts=self.opts, mod=task_emb, shape=self.layer.weight.shape)
            weight = weight * (1 - task_emb)
        return weight


class Binary_masking(torch.autograd.Function):
    """
    Binary masking object.
    """

    @staticmethod
    def forward(ctx, inputs: Tensor, DEFAULT_THRESHOLD: float, **kwargs):
        """

        Args:
            ctx: The ctx.
            inputs: The weight and the threshold.
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

    def __init__(self, opts: argparse, layer: nn.Module, mask: list, linear: bool):
        super(MaskWeight, self).__init__()
        self.weight = layer.weight
        self.mask_scale = 1e-2
        self.opts = opts
        self.ntasks = self.opts.ntasks
        self.masks = nn.ParameterList()
        self.default_threshold = opts.data_set_obj['threshold']
        if not linear:
            C1, C2, K1, K2 = layer.weight.shape
            m1, m2 = 1, 1
            new_shape = (K2, K1, C2 // m1, C1 // m2)
        else:
            C1, C2 = layer.weight.shape
            m1, m2 = opts.data_set_obj['mask_modulation_factor']
            new_shape = (C2 // m1, C1 // m2)

        self.channels = (C2, C1)
        for i in range(self.ntasks):
            layer = torch.zeros(new_shape)
            layer.fill_(self.mask_scale)
            layer = Parameter(layer)
            mask[i].append(layer)
            self.masks.append(layer)

    def forward(self, flags):
        """
        Forward the mask.
        Args:
            flags: The x.

        Returns: The masked conv.

        """
        task_id = flags[0]
        mask = self.masks[task_id]
        Mask = Binary_masking.apply(mask, self.default_threshold)
        Mask = Mask.view(self.weight.shape)
        new_weight = Mask * self.weight
        return new_weight


class Modulated_layer(nn.Module):
    """
    Conv with modulation and masking.
    """

    def __init__(self, opts: argparse, layer: nn.Module, task_embedding: list, create_modulation: bool,
                 create_masks: bool, masks: list, linear: bool, shape=None):
        """

        Args:
            opts: The Model opts.
            layer: The layer.
            task_embedding: The task embedding.
            create_modulation: Whether to create modulation.
            create_masks: Whether to create masks.
        """
        super(Modulated_layer, self).__init__()
        self.linear = linear
        self.opts = opts
        self.create_modulation = create_modulation
        self.create_masks = create_masks
        self.modulate_weights = opts.weight_modulation
        self.mask_weights = opts.mask_weights
        self.layer = layer
        if create_modulation:
            self.modulation: nn.Module = WeightModulation(opts=opts, layer=layer, modulations=task_embedding,
                                                          linear=linear, shape=shape)
        if create_masks:
            self.masks: nn.Module = MaskWeight(opts=opts, layer=layer, mask=masks, linear=linear)

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
        if self.linear:
            output = F.linear(input=x, weight=weights)
        else:
            output = F.conv2d(input=x, weight=weights, stride=self.layer.stride,
                              padding=self.layer.padding, dilation=self.layer.dilation, groups=self.layer.groups,
                              bias=self.layer.bias)
        return output
