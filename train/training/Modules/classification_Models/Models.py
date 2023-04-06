"""
Models part.
"""
import torch.nn as nn

from typing import Type, Union, Iterator

import torch
from torch import Tensor

from ..Module_Blocks import LambdaLayer, DownSample, WeightModulation, layer_with_modulation_and_masking, ChannelWiseTaskEmbedding
from .Blocks import BasicBlock, Bottleneck
from ..Batch_norm import BatchNormForClassification as BatchNorm
from .Heads import Head
import argparse

from ...Data.Structs import Spatial_Relations_inputs_to_struct

class ResNet(nn.Module):
    """
    ResNet model.
    """

    def __init__(
            self,
            opts: argparse,
            block: Type[Union[BasicBlock, Bottleneck]] = Bottleneck,
            num_blocks: list = [3, 4, 6, 3],
            channels: list = [64, 64, 128, 256, 512],
            ks: int = 7,
            pad: int = 3,
            strides: list = [2, 1, 2, 2, 2],
            use_max_pool: bool = True,
            groups: int = 1,
            width_per_group: int = 64,
    ) -> None:
        super(ResNet, self).__init__()
        self.opts = opts
        self.ntasks = opts.data_obj['ntasks']
        self.learnable_downsample = opts.data_obj['learnable_downsample']
        self.weight_modulation = opts.data_obj['weight_modulation']
        self.drop_out = opts.data_obj['drop_out_rate']
        self.heads = opts.data_obj['heads']
        self.modulations = [[] for _ in range(self.ntasks)]
        self.masks = [[] for _ in range(self.ntasks)]
        self.transfer_learning =  [[] for _ in range(self.ntasks)]
        self.num_blocks = num_blocks
        self.channels = channels
        self.kernel_size = ks
        self._norm_layer = BatchNorm
        self.inplanes = self.channels[0]
        self.strides = strides
        self.groups = groups
        self.base_width = width_per_group
        self.relu = nn.ReLU(inplace=True)
        if opts.arg_emb:
            self.arg = ChannelWiseTaskEmbedding(opts=opts)
        self.conv1 = nn.Conv2d(3, self.channels[0], kernel_size=self.kernel_size, stride=self.strides[0],
                               padding=pad, bias=False)
        self.bn1 = self._norm_layer(opts=self.opts, num_channels = self.channels[0])
        self.use_max_pool = use_max_pool
        if self.use_max_pool:
            self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layers = nn.ModuleList()
        for layer_id, num_layers in enumerate(self.num_blocks):
            layer = self._make_layer(block, self.channels[layer_id + 1], blocks=self.num_blocks[layer_id],
                                     stride=self.strides[layer_id + 1],
                                     modulation=self.modulations)
            self.layers.append(layer)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.drop_out = nn.Dropout(self.drop_out)
        self.linear = Head(self.channels[-1], heads= self.heads,
                 mask = self.masks, block_expansion = block.expansion,modulation=self.modulations)
        init_module_weights(self.modules())

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int, stride: int = 1,
                    modulation=None) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1:
            if self.learnable_downsample:
                downsample = DownSample(self.opts, self._norm_layer, self.inplanes, planes, block.expansion, stride)
            else:
                downsample = LambdaLayer(
                    lambda x: nn.functional.pad(x[:, :, ::2, ::2], [0, 0, 0, 0, planes // 4, planes // 4]),
                )
        elif self.inplanes != planes * block.expansion:
            downsample = DownSample(self.opts, self._norm_layer, self.inplanes, planes, block.expansion, stride)

        layers = [block(self.opts,
                        inplanes=self.inplanes, planes=planes, stride=stride, downsample=downsample,
                        norm_layer=norm_layer,
                        modulations=modulation,
                        masks=self.masks
                        )]

        self.inplanes = planes * block.expansion
        for idx in range(1, blocks):
            layer = block(self.opts,
                          self.inplanes,
                          planes,
                          norm_layer=norm_layer,

                          index=idx,
                          modulations=modulation, masks=self.masks)
            layers.append(layer)

        return nn.Sequential(*layers)

    def forward(self, samples:Spatial_Relations_inputs_to_struct) -> Tensor:
        """
        Forward of the model.
        Compute features and then the class probabilities.
        Args:
            x: The input.
            flags: The task flag.

        Returns: The class probabilities.

        """
        x, task_id = self.compute_feature(x=samples.image, flags=samples.task, arg_flag = samples.char_flags)
        x = self.linear(x, task_id)
        return x

    def compute_feature(self, x: Tensor, flags: Tensor,arg_flag:Tensor):
        """
        Compute the model features.
        Args:
            x: The input.
            flags: The flags.

        Returns: The model features.

        """
        x = self.arg(x, arg_flag)
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
        elif isinstance(m, (BatchNorm, nn.GroupNorm)):
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
        self.shapes = opts.data_set_obj.shapes
        self.layers = nn.ModuleList()
        self.ntasks = opts.data_set_obj.ntasks
        self.modulations = [[] for _ in range(self.ntasks)]
        self.masks = [[] for _ in range(self.ntasks)]
        self.relu = nn.ReLU(inplace=True)
        for idx, shape in enumerate(self.shapes[:-2]):
            layer = nn.Linear(self.shapes[idx], self.shapes[idx + 1], bias=False)
            layer = layer_with_modulation_and_masking(opts, layer, self.modulations, create_modulation=True,
                                                      create_masks=True, masks=self.masks, linear=True)
            self.layers.append(layer)
        self.head = nn.Linear(self.shapes[-2], self.shapes[-1])

    def forward(self, x: Tensor, task_id: int):
        """
        Compute the network output.
        Args:
            x: The input.
            task_id: The task id.

        Returns: The model output.

        """
        (B, _, _, _) = x.shape
        out = x.view(B, -1)
        for layer in self.layers:
            out = layer(out, task_id)
            out = self.relu(out)
        out = self.head(out)
        return out


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
    return ResNet(opts=opts, block=BasicBlock, num_blocks=[3, 3, 3], channels=[16, 32, 64, 64], strides=[2, 1, 2, 2],
                  pad=1)


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
    return ResNet(opts=opts, num_blocks=[3, 4, 6, 3], channels=[64, 64, 128, 256, 512], strides=[2, 1, 2, 2, 2])


def ResNet101(opts):
    """
       Original ResNet101 model.
       Args:
           opts: The model opts.

       Returns: ResNet101 model.

       """
    return ResNet(opts=opts, num_blocks=[3, 4, 23, 3])