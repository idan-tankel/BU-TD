"""
Here we create the basic blocks, for creation of the model.
"""
import torch.nn as nn

from typing import Callable, List, Tuple, Optional
import argparse

from torch import Tensor

from torchvision.models.resnet import conv3x3, conv1x1

from ..module_blocks import layer_with_modulation_and_masking


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
            masks: Optional[List] = None
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
        self.modulated_conv1 = layer_with_modulation_and_masking(opts=opts, layer=self.conv1,
                                                                 task_embedding=modulations, create_modulation=True,
                                                                 create_masks=True, masks=masks, linear=False)

        self.bn1 = norm_layer(planes, ntasks=self.ntasks)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.modulated_conv2 = layer_with_modulation_and_masking(opts=opts, layer=self.conv2,
                                                                 task_embedding=modulations, create_modulation=True,
                                                                 create_masks=True, masks=masks, linear=False)
        self.bn2 = norm_layer(planes, ntasks=self.ntasks)

        self.downsample = downsample
        self.option_B = opts.data_set_obj.option_B
        if self.downsample is not None and opts.data_set_obj.option_B:
            self.modulated_conv3 = layer_with_modulation_and_masking(opts=opts, layer=self.downsample.conv1x1,
                                                                     task_embedding=modulations, create_modulation=True,
                                                                     create_masks=True, masks=masks, linear=False)

        self.modulate_downsample = opts.data_set_obj.option_B and self.downsample is not None

    def forward(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Forward the block.
        Args:
            inputs: The block x.

        Returns: The block output.

        """
        identity, flags = inputs
        out = identity
        # If we modulate weight, we modulate and do forward.

        out = self.modulated_conv1(out, flags)

        out = self.bn1(out, flags)
        out = self.relu(out)
        # The same as above.
        out = self.modulated_conv2(out, flags)

        out = self.bn2(out, flags)
        if self.modulate_downsample:
            identity = self.modulated_conv3(identity, flags)
            identity = self.downsample.norm(identity, flags)

        elif self.downsample is not None:
            identity = self.downsample(identity, flags)
        out += identity
        out = self.relu(out)

        return out, flags


class Bottleneck(nn.Module):
    """
    Bottleneck block.
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
            modulations=None,
            masks=None
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
            inshapes: The block x shape.
            index: The block index.
        """
        super(Bottleneck, self).__init__()
        self.opts = opts
        self.ntasks = opts.data_set_obj['ntasks']
        self.modulations = []
        self.idx = index
        self.stride = stride
        self.inshapes = inshapes
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample num_blocks downsample the inputs when stride != 1
        self.use_mod1, self.use_mod2, self.use_mod3 = True, True, True
        self.conv1 = conv1x1(inplanes, width)
        self.modulated_conv1 = layer_with_modulation_and_masking(opts=opts, layer=self.conv1,
                                                                 task_embedding=modulations, create_modulation=True,
                                                                 create_masks=True, masks=masks,
                                                                 linear=False)
        self.bn1 = norm_layer(width, self.ntasks)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width, self.ntasks)
        self.modulated_conv2 = layer_with_modulation_and_masking(opts=opts, layer=self.conv2,
                                                                 task_embedding=modulations, create_modulation=True,
                                                                 create_masks=True, masks=masks,
                                                                 linear=False)

        self.conv3 = conv1x1(width, planes * self.expansion)
        self.modulated_conv3 = layer_with_modulation_and_masking(opts=opts, layer=self.conv3,
                                                                 task_embedding=modulations, create_modulation=True,
                                                                 create_masks=True, masks=masks,
                                                                 linear=False)
        self.bn3 = norm_layer(planes * self.expansion, self.ntasks)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        if self.downsample is not None:
            self.modulated_conv4 = layer_with_modulation_and_masking(opts=opts, layer=self.downsample.conv1x1,
                                                                     task_embedding=modulations, create_modulation=True,
                                                                     create_masks=True, masks=masks,
                                                                     linear=False)

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward the model.
        Args:
            inputs: The x.

        Returns: The model output.

        """
        identity, flags = inputs
        out = identity
        out = self.modulated_conv1(out, flags)

        out = self.bn1(out, flags)
        out = self.relu(out)
        out = self.modulated_conv2(out, flags)
        out = self.bn2(out, flags)
        out = self.relu(out)
        out = self.modulated_conv3(out, flags)
        out = self.bn3(out, flags)
        if self.downsample is not None:
            identity = self.modulated_conv4(identity, flags)
            identity = self.downsample.norm(identity, flags)

        out += identity
        out = self.relu(out)

        return out, flags
