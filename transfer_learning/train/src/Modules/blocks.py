"""
Here we create the basic blocks, for creation of the model.
"""
import torch.nn as nn

from typing import Callable, List, Tuple, Optional
import argparse

from torch import Tensor

from torchvision.models.resnet import conv3x3, conv1x1

import numpy as np

from src.Modules.module_blocks import WeightModulation, MaskWeight


class BasicBlock(nn.Module):
    """
    The basic block for the ResNet model.
    """
    expansion: int = 1

    def __init__(
            self,
            opts: argparse,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            inshapes: List = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            index: int = 0,
            modulations: Optional[List] = None,
            mask:Optional[List]
    ) -> None:
        """
         Create basic block with optional modulations.
        Args:
            opts: The model options.
            inplanes: The inplanes.
            planes: The outplanes.
            stride: The stride.
            downsample: The downsample layer.
            inshapes: The block inshape.
            norm_layer: The norm layer.
            index: The block index.
        """
        super(BasicBlock, self).__init__()
        self.shape = inshapes
        self.idx = index
        self.ntasks = opts.data_set_obj.ntasks
        self.stride = stride
        self.weight_modulation = opts.data_set_obj.weight_modulation
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.mask1 = MaskWeight(self.conv1)
        self.bn1 = norm_layer(planes, ntasks=self.ntasks)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.mask2 = MaskWeight(self.conv2)
        self.bn2 = norm_layer(planes, ntasks=self.ntasks)
        self.learn_mask = True
        # If we want to modulate the weights.
        if self.weight_modulation:
            self.modulation1 = WeightModulation(opts=opts, layer=self.conv1, modulations=modulations)
            self.modulation2 = WeightModulation(opts=opts, layer=self.conv2, modulations=modulations)
        # If we want to modulate the neurons.

        self.downsample = downsample
        if self.downsample is not None and opts.data_set_obj.option_B:
            self.modulation3 = WeightModulation(opts=opts, layer=self.downsample.conv1x1, modulations=modulations)
        self.modulate_downsample = self.weight_modulation and opts.data_set_obj.option_B

    def forward(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Forward the block.
        Args:
            inputs: The block input.

        Returns: The block output.

        """
        identity, flags = inputs
        out = identity
        # If we modulate weight, we modulate and do forward.
        if self.weight_modulation:
            out = self.modulation1(x=out, flags=flags)

        if self.learn_mask:
            out = self.mask1(out)
        # Else do ordinary forward.
        else:
            out = self.conv1(out)

        out = self.bn1(out, flags)
        out = self.relu(out)
        # The same as above.
        if self.weight_modulation:
            out = self.modulation2(x=out, flags=flags)

        if self.learn_mask:
            out = self.mask2(out)
        else:
            out = self.conv2(out)

        out = self.bn2(out, flags)
        if self.downsample is not None and self.modulate_downsample:
            if self.weight_modulation:
                identity = self.modulation3(identity, flags)
                identity = self.downsample.norm(identity, flags)
            else:
                identity = self.downsample(identity, flags)
        elif self.downsample is not None:
            identity = self.downsample(identity, flags)
        out += identity
        out = self.relu(out)

        return out, flags


class Bottleneck(nn.Module):
    """
    # Bottleneck in torchvision places the stride for down-sampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    """
    expansion: int = 4

    def __init__(
            self,
            opts: argparse,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            inshapes: Optional[List] = None,
            index: int = 0,
            modulations=None

    ) -> None:
        """
        Create basic block with optional modulations.
        Args:
            opts: The model opts.
            inplanes: The block inplanes.
            planes: The block outplanes.
            stride: The stride for first block.
            downsample: The downsample operator.
            groups: The groups.
            base_width: The base width.
            dilation: The dilation
            norm_layer: The batch norm layer.
            inshapes: The block input shape.
            index: The block index.
        """
        super(Bottleneck, self).__init__()
        self.opts = opts
        self.ntasks = opts.data_set_obj.ntasks
        self.weight_modulation = opts.data_set_obj.weight_modulation
        self.use_neural_modulation = opts.use_neural_modulation
        self.modulations = []
        self.idx = index
        self.inshapes = inshapes
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample num_blocks downsample the inputs when stride != 1
        c, h, w = inshapes
        self.use_mod1, self.use_mod2, self.use_mod3 = True, True, True
        self.conv1 = conv1x1(inplanes, width)
        if self.use_mod1:
            self.mod1 = WeightModulation(opts=opts, layer=self.conv1, modulations=modulations)
        if self.use_neural_modulation:
            self.neural_mod1 = NeuralModulation(opts, shape=self.inshapes, block_exp=self.expansion,
                                                modulations=modulations)
        self.bn1 = norm_layer(width, self.ntasks)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width, self.ntasks)
        new_shape = [width, np.int(np.ceil(h / stride)), np.int(np.ceil(w / stride))]
        if self.use_mod2:
            self.mod2 = WeightModulation(opts=opts, layer=self.conv2, modulations=modulations)
        if self.use_neural_modulation:
            self.neural_mod2 = NeuralModulation(opts, shape=self.inshapes, block_exp=self.expansion,
                                                modulations=modulations)

        self.conv3 = conv1x1(width, planes * self.expansion)
        if self.use_mod3:
            self.mod3 = WeightModulation(opts=opts, layer=self.conv3, modulations=modulations)
        if self.use_neural_modulation:
            self.neural_mod3 = NeuralModulation(opts, shape=self.inshapes, block_exp=self.expansion,
                                                modulations=modulations)
        c, h, w = new_shape
        self.bn3 = norm_layer(planes * self.expansion, self.ntasks)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        if self.use_mod2 and self.downsample is not None and self.opts.data_set_obj.option_B:
            self.mod4 = WeightModulation(opts=opts, layer=self.downsample.conv1x1, modulations=modulations)

        self.stride = stride
        new_shape = [planes * self.expansion, np.int(np.ceil(h)), np.int(np.ceil(w))]
        self.shape = new_shape
        self.modulate_downsample = self.weight_modulation and opts.data_set_obj.option_B

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward the model.
        Args:
            inputs: The input.

        Returns: The model output.

        """
        identity, flags = inputs
        out = identity
        if self.use_mod1:
            out = self.mod1(out, flags)
        else:
            out = self.conv1(out)

        out = self.bn1(out, flags)
        out = self.relu(out)
        if self.use_neural_modulation:
            out = self.neural_mod1(out, flags)
        if self.use_mod2:
            out = self.mod2(out, flags)
        else:
            out = self.conv2(out)
        out = self.bn2(out, flags)
        out = self.relu(out)
        if self.use_neural_modulation:
            out = self.neural_mod2(out, flags)
        if self.use_mod3:
            out = self.mod3(out, flags)
        else:
            out = self.conv3(out)
        out = self.bn3(out, flags)
        if self.use_neural_modulation:
            out = self.neural_mod3(out, flags)
        if self.downsample is not None:
            if self.modulate_downsample:
                identity = self.mod4(identity, flags)
                identity = self.downsample.norm(identity, flags)
            else:
                identity = self.downsample(identity, flags)

        out += identity
        out = self.relu(out)

        return out, flags
