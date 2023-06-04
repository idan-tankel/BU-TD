import copy
import math

from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn, Tensor

from ...continual_learning_layers.Heads import Multi_task_head

from .blocks import Conv2dNormActivationOurs, MBConvConfigOurs
from ...continual_learning_layers.module_blocks import WeightModulation, Modulated_layer


def _efficientnet_conf(
        arch: str,
        **kwargs: Any,
) -> Tuple[Sequence[Union[MBConvConfigOurs]], Optional[int]]:
    inverted_residual_setting: Sequence[Union[MBConvConfigOurs]]
    if arch.startswith("efficientnet_b"):
        bneck_conf = partial(MBConvConfigOurs, width_mult=kwargs.pop("width_mult"), depth_mult=kwargs.pop("depth_mult"))
        inverted_residual_setting = [
            bneck_conf(1, 3, 1, 32, 16, 1),
            bneck_conf(6, 3, 2, 16, 24, 2),
            bneck_conf(6, 5, 2, 24, 40, 2),
            bneck_conf(6, 3, 2, 40, 80, 3),
            bneck_conf(6, 5, 1, 80, 112, 3),
            bneck_conf(6, 5, 2, 112, 192, 4),
            bneck_conf(6, 3, 1, 192, 320, 1),
        ]
        last_channel = None
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, last_channel


class EfficientNet(nn.Module):
    def __init__(
            self,
            opts,
            inverted_residual_setting: Sequence[Union[MBConvConfigOurs]],
            stochastic_depth_prob: float = 0.2,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            last_channel: Optional[int] = None,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        self.opts = opts
        self.ntasks = opts.data_set_obj['ntasks']
        self.masks = [[] for _ in range(self.ntasks)]
        self.modulations = [[] for _ in range(self.ntasks)]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []

        # building first layer
        first_conv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivationOurs(opts=opts, masks=self.masks, modulation=self.modulations,
                                     input_channels=3, expanded_channels=first_conv_output_channels, kernel_size=3,
                                     stride=2,
                                     norm_layer=norm_layer,
                                     activation_layer=nn.SiLU
                                     )
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        self.inverted_residual_setting = inverted_residual_setting
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block_cnf.block(opts=opts, masking=self.masks, modulation=self.modulations, cnf=block_cnf,
                                             stochastic_depth_prob=sd_prob,
                                             norm_layer=norm_layer))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        # building last several layers
        last_conv_input_channels = inverted_residual_setting[-1].out_channels
        last_conv_output_channels = last_channel if last_channel is not None else 4 * last_conv_input_channels
        layers.append(
            Conv2dNormActivationOurs(
                opts=opts,
                modulation=self.modulations,
                masks=self.masks,
                input_channels=last_conv_input_channels,
                expanded_channels=last_conv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.drop_out = nn.Dropout(opts.data_set_obj['drop_out_rate'])
        num_classes = opts.data_set_obj['heads']
        self.classifier = Multi_task_head(in_channels=last_conv_output_channels, heads=num_classes, modulation=self.modulations,
                               mask=self.masks)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)
            if isinstance(m, WeightModulation):
                for param in m.modulation:
                    nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def _forward_impl(self, x: Tensor, flags: Tensor) -> Tensor:
        out = (x, flags)
        for layer in self.features:
            out = layer(out)
        out, _ = out
        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        out = self.drop_out(out)

        out = self.classifier(out, flags)

        return out

    def forward(self, x: Tensor, flags: Tensor) -> Tensor:
        return self._forward_impl(x=x, flags=flags)


def _efficientnet(
        opts,
        inverted_residual_setting: Sequence[Union[MBConvConfigOurs]],
        last_channel: Optional[int],
        **kwargs: Any,
) -> EfficientNet:
    model = EfficientNet(opts=opts, inverted_residual_setting=inverted_residual_setting,
                         last_channel=last_channel, **kwargs)

    return model


def efficientnet_b0(
        *, opts, **kwargs: Any
) -> EfficientNet:
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b0", width_mult=1.0, depth_mult=1.0)
    return _efficientnet(opts=opts, inverted_residual_setting=inverted_residual_setting, last_channel=last_channel,
                         **kwargs)
