from typing import Callable, List, Optional, Tuple
import argparse
import torch
from torch import nn, Tensor

from torchvision.models.mobilenetv2 import MobileNet_V2_Weights, _log_api_usage_once, \
    _make_divisible

from .Blocks import ConvNormActivation as Conv2dNormActivation

from ...continual_learning_layers.Heads import Multi_task_head

from ....Utils import num_params

__all__ = ["MobileNetV2", "MobileNet_V2_Weights", "Conv2dNormActivation"]

from ...continual_learning_layers.module_blocks import Modulated_layer, WeightModulation

from ...continual_learning_layers.Batch_norm import BatchNorm

from ....data.Enums import TrainingFlag

from ...continual_learning_layers.Heads import Multi_task_head


# TODO - CHANGE THE WAY LAYERS ARE USED IN FORWARD.

# necessary for backwards compatibility
class InvertedResidual(nn.Module):
    def __init__(
            self, opts: argparse, inp: int, oup: int, stride: int, expand_ratio: int,
            masks: List, modulation: List, norm_layer: Optional[Callable[...,
            nn.Module]]
            = None
    ) -> None:
        super().__init__()
        training_flag = opts.training_flag
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup

        layers: List[nn.Module] = []
        self.expand_ratio = expand_ratio
        if expand_ratio != 1:
            # pw
            self.first_layer = Conv2dNormActivation(opts, inp, hidden_dim, kernel_size=1, norm_layer=norm_layer,
                                                    activation_layer=nn.ReLU6)
            self.first_seq = nn.Sequential(self.first_layer.conv, self.first_layer.norm_layer,
                                           self.first_layer.activation)
            self.first_bn = self.first_layer.norm_layer
            self.first_activation = self.first_layer.activation
            layers.append(self.first_seq)
            self.first_conv = Modulated_layer(opts=opts, layer=self.first_layer.conv,
                                              create_modulation=training_flag is TrainingFlag.Modulation,
                                              create_masks=training_flag is TrainingFlag.Masks, linear=False,
                                              masks=masks,
                                              task_embedding=modulation)

        layer1 = Conv2dNormActivation(opts,
                                      hidden_dim,
                                      hidden_dim,
                                      stride=stride,
                                      groups=hidden_dim,
                                      norm_layer=norm_layer,
                                      activation_layer=nn.ReLU6,
                                      )

        layer1_seq = nn.Sequential(layer1.conv, layer1.norm_layer, layer1.activation)

        self.second_conv = Modulated_layer(opts=opts, layer=layer1.conv, create_modulation=training_flag is
                                                                                           TrainingFlag.Modulation,
                                           create_masks=training_flag is TrainingFlag.Masks, linear=False, masks=masks,
                                           task_embedding=modulation)
        self.second_bn = layer1.norm_layer
        self.second_activation = layer1.activation

        layer2 = nn.Conv2d(hidden_dim, oup, 1, bias=False)
        self.third_conv = Modulated_layer(opts=opts, layer=layer2,
                                          create_modulation=training_flag is TrainingFlag.Modulation,
                                          create_masks=training_flag is TrainingFlag.Masks, linear=False, masks=masks,
                                          task_embedding=modulation)

        self.third_norm = norm_layer(oup, opts.ntasks)

        layers.extend(
            [
                # dw
                layer1_seq,
                # pw-linear
                layer2,
                self.third_norm
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
            out = self.first_conv(out, flags)
            out = self.first_bn((out, flags))
            out = self.first_activation(out)
        out = self.second_conv(out, flags)
        out = self.second_bn((out, flags))
        out = self.second_activation(out)
        out = self.third_conv(out, flags)
        out = self.third_norm((out, flags))

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
            round_nearest: int = 8
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
        self.training_flag = opts.training_flag
        self.ntasks = opts.ntasks
        self.masks = [[] for _ in range(opts.ntasks)]
        self.modulations = [[] for _ in range(opts.ntasks)]

        self.block = InvertedResidual

        self.norm_layer = BatchNorm

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
            raise ValueError(f"inverted_residual_setting should be non-empty or"
                             f" a 4-element list, got {inverted_residual_setting}")

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        first_layer = Conv2dNormActivation(opts, 3, input_channel, stride=2, norm_layer=self.norm_layer,
                                           activation_layer=nn.ReLU6)
        self.conv1 = Modulated_layer(opts=opts, layer=first_layer.conv, masks=self.masks,
                                     task_embedding=self.modulations,
                                     linear=False, create_masks=self.training_flag is TrainingFlag.Masks,
                                     create_modulation=self.training_flag is TrainingFlag.Modulation)

        self.first_bn = first_layer.norm_layer

        self.first_activation = first_layer.activation

        first_seq = nn.Sequential(first_layer.conv, first_layer.norm_layer, first_layer.activation)

        features: List[nn.Module] = [first_seq]
        self.body_layers = nn.Sequential()
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                layer = self.block(opts=opts, inp=input_channel, oup=output_channel, stride=stride,
                                   expand_ratio=t, norm_layer=self.norm_layer, masks=self.masks,
                                   modulation=self.modulations)
                features.append(layer)
                self.body_layers.append(layer)
                input_channel = output_channel
        # building last several layers

        last_layer = Conv2dNormActivation(opts, input_channel, self.last_channel, kernel_size=1, norm_layer=self.norm_layer,
                                          activation_layer=nn.ReLU6)

        self.last_conv = Modulated_layer(layer=last_layer.conv, opts=opts, masks=self.masks,
                                         task_embedding=self.modulations, linear=False,
                                         create_modulation=self.training_flag is TrainingFlag.Modulation,
                                         create_masks=self.training_flag is TrainingFlag.Masks)

        self.last_bn = last_layer.norm_layer

        self.last_relu = last_layer.activation

        last_seq = nn.Sequential(last_layer.conv, last_layer.norm_layer, last_layer.activation)

        features.append(last_seq)
        # make it nn.Sequential

        self.features = nn.Sequential(*features)

        # building classifier

        self.classifier = Multi_task_head(opts=opts, modulation=self.modulations, mask=self.masks,
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
        The Model forward.
        Args:
            x: The input.
            flags: The flags.

        Returns: The Model output.

        """
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.conv1(x, flags)
        x = self.first_bn((x, flags))
        x = self.first_activation(x)
        x, _ = self.body_layers((x, flags))
        x = self.last_conv(x, flags)
        x = self.last_bn((x, flags))
        x = self.last_relu(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x
