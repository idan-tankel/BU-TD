from typing import Callable, List, Optional, Tuple
import argparse
import torch
from torch import nn, Tensor

from torchvision.models.mobilenetv2 import MobileNet_V2_Weights, _log_api_usage_once, \
    _make_divisible

from .Blocks import ConvNormActivation as Conv2dNormActivation

from ..Heads import Head

__all__ = ["MobileNetV2", "MobileNet_V2_Weights", "Conv2dNormActivation"]

from ..module_blocks import layer_with_modulation_and_masking, WeightModulation

from ..Batch_norm import BatchNorm

# necessary for backwards compatibility
class InvertedResidual(nn.Module):
    def __init__(
            self, opts: argparse, inp: int, oup: int, stride: int, expand_ratio: int,
            masks: List, modulation: List, norm_layer: Optional[Callable[...,
            nn.Module]]
            = None
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        self.expand_ratio = expand_ratio
        if expand_ratio != 1:
            # pw
            layers.append(
                Conv2dNormActivation(opts, inp, hidden_dim, kernel_size=1, norm_layer=norm_layer,
                                     activation_layer=nn.ReLU6)
            )
            self.layer1 = layer_with_modulation_and_masking(opts=opts, layer=layers[0][0], create_modulation=True,
                                                            create_masks=True, linear=False, masks=masks,
                                                            task_embedding=modulation)

        layer1 = Conv2dNormActivation(opts,
            hidden_dim,
            hidden_dim,
            stride=stride,
            groups=hidden_dim,
            norm_layer=norm_layer,
            activation_layer=nn.ReLU6,
        )
        self.layer2 = layer_with_modulation_and_masking(opts=opts, layer=layer1[0], create_modulation=True,
                                                        create_masks=True, linear=False, masks=masks,
                                                        task_embedding=modulation)

        layer2 = nn.Conv2d(hidden_dim, oup, 1, bias=False)
        self.layer3 = layer_with_modulation_and_masking(opts=opts, layer=layer2, create_modulation=True,
                                                        create_masks=True, linear=False, masks=masks,
                                                        task_embedding=modulation)

        layers.extend(
            [
                # dw
                layer1,
                # pw-linear
                layer2,
                norm_layer(oup,opts.ntasks),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """
        The forward.
        Args:
            inputs: The input, flags.

        Returns: The output, flags.

        """
        x, flags = inputs
        out = x
        if self.expand_ratio != 1:
            out = self.layer1(out, flags)
            out = self.conv[0][1](out)
            out = self.conv[0][2](out)
        out = self.layer2(out, flags)
        out = self.conv[-3][1](out)
        out = self.conv[-3][2](out)
        out = self.layer3(out, flags)
        out = self.conv[-1](out)

        if self.use_res_connect:
            return x + out, flags
        else:
            return out, flags


class MobileNetV2(nn.Module):
    """
    MobileNet.
    """

    def __init__(
            self,
            opts: argparse,
            width_mult: float = 1.0,
            inverted_residual_setting: Optional[List[List[int]]] = None,
            round_nearest: int = 8,
            block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        """
        MobileNet V2 main class

        Args:
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super().__init__()
        _log_api_usage_once(self)
        self.masks = [[] for _ in range(opts.ntasks)]
        self.modulations = [[] for _ in range(opts.ntasks)]

        if block is None:
            block = InvertedResidual

        norm_layer = BatchNorm

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
            )

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        layer = Conv2dNormActivation(opts, 3, input_channel, stride=2, norm_layer=norm_layer, activation_layer=nn.ReLU6)
        self.conv1 = layer_with_modulation_and_masking(opts=opts, layer=layer[0], masks=self.masks,
                                                       task_embedding=self.modulations,
                                                       linear=False, create_masks=True, create_modulation=True)
        self.bn1 = layer[1]

        self.relu1 = layer[2]
        features: List[nn.Module] = [layer]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(opts=opts, inp=input_channel, oup=output_channel, stride=stride, expand_ratio=t,
                                      norm_layer=norm_layer, masks=self.masks, modulation=self.modulations
                                      ))
                input_channel = output_channel
        # building last several layers
        #
        layer = Conv2dNormActivation(opts,
            input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6)
        self.last_conv = layer_with_modulation_and_masking(layer=layer[0], opts=opts, masks=self.masks,
                                                           task_embedding=self.modulations, linear=False,
                                                           create_modulation=True, create_masks=True)
        self.last_bn = layer[1]
        self.last_relu = layer[2]
        features.append(layer)
        # make it nn.Sequential
        self.features = nn.Sequential(*features)
        # building classifier

        self.classifier = Head(opts=opts, modulation=self.modulations, mask=self.masks,
                               in_channels=self.last_channel)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
            if isinstance(m, WeightModulation):
                if m.modulation is not None:
                    for param in m.modulation:
                        for sub_param in param:
                            if 0 not in sub_param.shape:
                                nn.init.kaiming_normal_(sub_param, mode='fan_out', nonlinearity='relu')

    def forward(self, x: Tensor, flags: Tensor) -> Tensor:
        """
        The model forward.
        Args:
            x: The input.
            flags: The flags.

        Returns: The model output.

        """
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.conv1(x, flags)
        x = self.bn1(x)
        x = self.relu1(x)
        x, _ = self.features[1:len(self.features) - 1]((x, flags))
        x = self.last_conv(x, flags)
        x = self.last_bn(x)
        x = self.last_relu(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x


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
            x: The x.
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
