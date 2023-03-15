"""
Models part.
"""
import torch.nn as nn
import numpy as np
from typing import Type, Union, Iterator

import torch
from torch import Tensor

from src.Modules.module_blocks import LambdaLayer, DownSample, WeightModulation
from src.Modules.blocks import BasicBlock, Bottleneck
from src.Modules.Batch_norm import BatchNorm as Batch_Norm_ours_with_saving_stats
from src.Modules.Heads import Head
from src.Utils import Expand


class ResNet(nn.Module):
    """
    ResNet model.
    """

    def __init__(
            self,
            opts,
            block,
            num_blocks,
            groups: int = 1,
            width_per_group: int = 64,
    ) -> None:
        super(ResNet, self).__init__()
        self.opts = opts
        self.num_blocks = num_blocks
        self.channels = opts.data_set_obj.channels
        self.ntasks = opts.data_set_obj.ntasks
        self.kernel_size = opts.data_set_obj.ks
        self.modulations = [[] for _ in range(self.ntasks)]
        self.masks = [[] for _ in range(self.ntasks)]
        self.option_B = opts.data_set_obj.option_B
        self._norm_layer = Batch_Norm_ours_with_saving_stats
        self.image_shape = opts.data_set_obj.image_shape
        self.inplanes = self.channels[0]
        self.strides = opts.data_set_obj.strides
        self.groups = groups
        self.base_width = width_per_group
        self.weight_modulation = opts.data_set_obj.weight_modulation
        self.relu = nn.ReLU(inplace=True)
        self.initial_pad = opts.data_set_obj.pad
        self.drop_out = opts.data_set_obj.drop_out_rate
        self.heads = opts.data_set_obj.heads
        self.conv1 = nn.Conv2d(3, self.channels[0], kernel_size=self.kernel_size[0], stride=self.strides[0],
                               padding=self.initial_pad, bias=False)
        self.bn1 = self._norm_layer(self.channels[0], ntasks=self.ntasks)
        c, h, w = opts.data_set_obj.image_shape
        shape = [self.channels[0], np.int(np.ceil(h / 2)), np.int(np.ceil(w / 2))]
        self.use_max_pool = opts.data_set_obj.use_max_pool
        if self.use_max_pool:
            self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        c, h, w = shape
        shape = [self.channels[0], np.int(np.ceil(h / self.strides[0])), np.int(np.ceil(w / self.strides[0]))]
        self.layers = nn.ModuleList()
        for layer_id, num_layers in enumerate(self.num_blocks):
            c, h, w = shape
            shape = [self.channels[layer_id + 1], np.int(np.ceil(h / self.strides[layer_id + 1])),
                     np.int(np.ceil(w / self.strides[layer_id + 1]))]
            layer = self._make_layer(block, self.channels[layer_id + 1], blocks=self.num_blocks[layer_id], shape=shape,
                                     stride=self.strides[layer_id + 1],
                                     modulation=self.modulations)
            self.layers.append(layer)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.drop_out = nn.Dropout(self.drop_out)
        self.linear = Head(in_channels=self.channels[-1], heads=self.heads,
                           modulation=self.modulations, block_expansion=block.expansion, mask=self.masks)
        init_module_weights(self.modules())

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int, stride: int = 1,
                    shape=None, modulation=None) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1:
            if self.option_B:
                downsample = DownSample(self._norm_layer, self.inplanes, planes, block.expansion, stride, self.ntasks)
            else:
                downsample = LambdaLayer(
                    lambda x: nn.functional.pad(x[:, :, ::2, ::2], [0, 0, 0, 0, planes // 4, planes // 4]),
                )
        elif self.inplanes != planes * block.expansion:
            downsample = DownSample(self._norm_layer, self.inplanes, planes, block.expansion, stride, self.ntasks)

        layers = [block(self.opts,
                        inplanes=self.inplanes, planes=planes, stride=stride, downsample=downsample,
                        norm_layer=norm_layer, inshapes=shape,
                        modulations=modulation,
                        masks=self.masks
                        )]

        shape = layers[0].shape
        self.inplanes = planes * block.expansion
        for idx in range(1, blocks):
            layer = block(self.opts,
                          self.inplanes,
                          planes,
                          norm_layer=norm_layer,
                          inshapes=shape,
                          index=idx,
                          modulations=modulation, masks=self.masks)
            layers.append(layer)

        return nn.Sequential(*layers)

    def forward(self, x: Tensor, flags: Tensor) -> Tensor:
        """
        Forward of the model.
        Compute features and then the class probabilities.
        Args:
            x: The input.
            flags: The task flag.

        Returns: The class probabilities.

        """
        x, task_id = self.compute_feature(x=x, flags=flags)
        x = self.linear(x, task_id)
        return x

    def compute_feature(self, x: Tensor, flags: Tensor):
        """
        Compute the model features.
        Args:
            x: The input.
            flags: The flags.

        Returns: The model features.

        """
        x: Tensor = self.conv1(x)
        x: Tensor = self.bn1(x, flags)
        x: Tensor = self.relu(x)
        if self.use_max_pool:
            x = self.max_pool(x)
        inputs = (x, flags)
        for layer in self.layers:
            inputs = layer(inputs)

        inputs, task_id = inputs
        inputs = self.avgpool(inputs)
        inputs = self.drop_out(inputs)
        inputs = torch.flatten(inputs, 1)
        return inputs, task_id


def init_module_weights(modules: Iterator[nn.Module]) -> None:
    """
    Initialize the modules.
    Args:
        modules: The model modules.

    Returns: None

    """
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, (Batch_Norm_ours_with_saving_stats, nn.GroupNorm)):
            nn.init.constant_(m.norm.weight, 1)
            nn.init.constant_(m.norm.bias, 0)

        if isinstance(m, WeightModulation):
            for param in m.modulation:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")


class SimpleMLP(nn.Module):
    """
    Simple MLP.
    """

    def __init__(self, opts):
        super(SimpleMLP, self).__init__()
        in_shape, in_channels, num_classes = opts.data_set_obj.in_shape, \
            opts.data_set_obj.in_channels, opts.data_set_obj.num_classes
        self.in_channels = in_channels
        self.net = nn.Sequential(nn.Linear(in_shape, in_channels, bias=False), nn.ReLU(),
                                 nn.Linear(in_channels, num_classes, bias=False))
        size = (in_channels // 5, in_shape // 2)
        self.mod1 = nn.Parameter(torch.Tensor(*size), requires_grad=True)
        nn.init.xavier_uniform_(self.mod1)
        size = (num_classes // 2, in_channels // 5)
        self.mod2 = nn.Parameter(torch.Tensor(*size), requires_grad=True)
        nn.init.xavier_uniform_(self.mod2)
        self.use_mod = False
        self.modulations = [self.mod1, self.mod2]
        self.linear = self.net[-1]

    def forward(self, x: Tensor, task_id: int):
        """
        Compute the network output.
        Args:
            x: The input.
            task_id: The task id.

        Returns: The model output.

        """
        (B, _, _, _) = x.shape
        x = x.view(B, -1)
        weight = self.net[0].weight
        if self.use_mod:
            weight = self.net[0].weight * Expand(self.mod1, shape=[5, 2])
        x = nn.functional.linear(x, weight=weight, bias=None)
        x = self.net[2](x)
        weight = self.net[2].weight
        if self.use_mod:
            weight = self.net[2].weight * Expand(self.mod2, shape=[2, 5])
        x = nn.functional.linear(x, weight=weight, bias=None)
        return x


class ExpandedSimpleMLP(nn.Module):
    """
    Baseline expanded MLP.
    """

    def __init__(self, in_shape, in_channels, num_classes):
        super(ExpandedSimpleMLP, self).__init__()
        self.in_channels = in_channels
        size = (in_channels // 5, in_shape // 2)
        self.mod1 = nn.Parameter(torch.Tensor(*size), requires_grad=True)
        nn.init.xavier_uniform_(self.mod1)
        size = (num_classes // 2, in_channels // 5)
        self.mod2 = nn.Parameter(torch.Tensor(*size), requires_grad=True)
        nn.init.xavier_uniform_(self.mod2)
        self.use_mod = True
        self.modulations = [self.mod1, self.mod2]

    def forward(self, x: Tensor, task_id: int):
        """
        Compute the network output.
        Args:
            x: The input.
            task_id: The task id.

        Returns: The model output.

        """
        (B, _, _, _) = x.shape
        x = x.view(B, -1)
        weight = Expand(self.mod1, shape=[5, 2])
        x = nn.functional.linear(x, weight=weight, bias=None)
        x = nn.ReLU()(x)
        weight = Expand(self.mod2, shape=[2, 5])
        x = nn.functional.linear(x, weight=weight, bias=None)
        return x


def ResNet14(opts):
    """
    Small ResNet32 model with more blocks but with low number of filters.
     Args:
        opts: The model opts.

    Returns: ResNet32 model.

    """
    return ResNet(opts=opts, block=BasicBlock, num_blocks=[2, 2, 2])


def ResNet18(opts):
    """
    Original ResNet18 model.
    Args:
        opts: The model opts.

    Returns: ResNet18 model.

    """
    return ResNet(opts=opts, block=BasicBlock, num_blocks=[2, 2, 2, 2])


def ResNet20(opts):
    """
    Small ResNet32 model with more blocks but with low number of filters.
     Args:
        opts: The model opts.

    Returns: ResNet32 model.

    """
    return ResNet(opts=opts, block=BasicBlock, num_blocks=[3, 3, 3])


def ResNet32(opts):
    """
    Small ResNet32 model with more blocks but with low number of filters.
     Args:
        opts: The model opts.

    Returns: ResNet32 model.

    """
    return ResNet(opts=opts, block=BasicBlock, num_blocks=[5, 5, 5])


def ResNet34(opts):
    """
       Original ResNet34 model.
       Args:
           opts: The model opts.

       Returns: ResNet34 model.

       """
    return ResNet(opts=opts, block=BasicBlock, num_blocks=[3, 4, 6, 3])


def ResNet50(opts):
    """
       Original ResNet50 model.
       Args:
           opts: The model opts.

       Returns: ResNet50 model.

       """
    return ResNet(opts=opts, block=Bottleneck, num_blocks=[3, 4, 6, 3])


def ResNet101(opts):
    """
       Original ResNet101 model.
       Args:
           opts: The model opts.

       Returns: ResNet101 model.

       """
    return ResNet(opts=opts, block=Bottleneck, num_blocks=[3, 4, 23, 3])
