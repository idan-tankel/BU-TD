"""
Module blocks.
"""
import torch.nn as nn

from typing import List
import argparse

from torch import Tensor

import torch

import torch.nn.functional as f

from ...Utils import expand, compute_size


class ModulateWeights(nn.Module):
    def __init__(self, opts: argparse, layer: nn.Module, size: List, list_params):
        """

        Args:
            opts: The model options.
            layer: The current later to modulate its weight.
        """
        super(ModulateWeights, self).__init__()
        self.opts = opts
        self.num_tasks = opts.General_specification['num_tasks']
        self.layer = layer
        self.training_type = opts.training_type
        self.modulation = nn.ParameterList()
        for i in range(self.num_tasks):
            param = nn.Parameter(torch.Tensor(*size))
            self.modulation.append(param)
            list_params[i].append(self.modulation[i])

    def forward(self, modulation):
        """
        Compute the forward modulation pass.
        Args:
            modulation: The weight modulation.

        Returns: The modulated weights.

        """
        return self.layer.weight * modulation


class WeightModulation(ModulateWeights):
    """
    NeuralModulation layer.
    Create Channel or Column modulations layer.
    The main idea of the paper allowing continual learning without forgetting.
    """

    def __init__(self, opts: argparse, layer: nn.Module, modulations: List):
        """

        Args:
            opts: The model options.
            layer: The current later to modulate its weight.
        """

        size = compute_size(layer.weight.shape, opts.task_specification['weight_modulation_factor'])
        super(WeightModulation, self).__init__(opts=opts, layer=layer, size=size, list_params=modulations)

    def forward(self, tasks: Tensor) -> Tensor:
        """
        Perform the weights modulations.
        Args:
            tasks: The task id.

        Returns: Torch of shape [B,C,H,W], the modulated tensor.

        """
        task_id = tasks[0]
        if self.modulation[task_id].size(0) > 0:
            task_emb = expand(self.opts, self.modulation[task_id], shape=self.layer.weight.shape)
            return super().forward(modulation = 1 - task_emb)
        else:
            return self.layer.weight


class BinaryMasking(torch.autograd.Function):
    """
    Binary masking object.
    """

    @staticmethod
    def forward(ctx, inputs, default_threshold):
        """

        Args:
            ctx: The ctx.
            inputs: The weight and the threshold.
            default_threshold: The threshold.

        Returns:

        """
        outputs = inputs.clone()
        outputs[inputs.le(default_threshold)] = 0
        outputs[inputs.gt(default_threshold)] = 1
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


class MaskWeight(ModulateWeights):
    """
    Mask weight.
    """

    def __init__(self, opts: argparse, layer: nn.Module, mask: list):
        self.opts = opts
        self.default_threshold = opts.task_specification['threshold']
        new_shape = layer.weight.shape
        super(MaskWeight, self).__init__(opts=opts, list_params=mask, size=new_shape, layer=layer)

    def forward(self, tasks):
        """
        Forward the mask.
        Args:
            tasks: The task ids.

        Returns: The masked conv.

        """
        task_id = tasks[0]
        mask = self.modulation[task_id]
        mask = BinaryMasking.apply(mask, self.default_threshold)
        return super().forward(modulation=mask)


class LayerWithModulationAndMasking(nn.Module):
    """
    Conv with modulation and masking.
    """

    def __init__(self, opts: argparse, layer: nn.Module, task_embedding: list, create_modulation: bool,
                 create_masks: bool, masks: list):
        """

        Args:
            opts: The model opts.
            layer: The layer.
            task_embedding: The task embedding.
            create_modulation: Whether to create modulation.
            create_masks: Whether to create masks.
        """
        super(LayerWithModulationAndMasking, self).__init__()
        self.opts = opts
        self.create_modulation = create_modulation
        self.create_masks = create_masks
        self.modulate_weights = opts.weight_modulation
        self.mask_weights = opts.mask_weights
        self.layer = layer
        if self.modulate_weights:
            self.modulation: nn.Module = WeightModulation(opts=opts, layer=layer, modulations=task_embedding)
        if self.mask_weights:
            self.masks: nn.Module = MaskWeight(opts=opts, layer=layer, mask=masks)

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
            weights = self.modulation(tasks=flags)
        if self.create_masks and self.mask_weights:
            weights = self.masks(tasks=flags)

        output = f.conv2d(input=x, weight=weights, stride=self.layer.stride,
                          padding=self.layer.padding, dilation=self.layer.dilation, groups=self.layer.groups,
                          bias=self.layer.bias)
        return output
