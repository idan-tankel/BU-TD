import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def flag_to_task(flag: torch) -> int:
    """
    Args:
        flag: The One hot flag.

    Returns: The id in which the flag == 1.

    """
    task = torch.argmax(flag, dim=1)[0]  # Finds the non zero entry in the one-hot vector
    return task


def get_laterals(laterals: list[torch], layer_id: int, block_id: int) -> torch:
    """
    Returns the lateral connections associated with the layer, block.
    Args:
        laterals: All lateral connections from the previous stream, if exists.
        layer_id: The layer id in the stream.
        block_id: The block id in the layer.

    Returns: All the lateral connections associate with the block(usually 3).

    """
    if laterals is None:  # If BU1, there are not any lateral connections.
        return None
    if len(laterals) > layer_id:  # assert we access to an existing layer.
        layer_laterals = laterals[layer_id]  # Get all lateral associate with the layer.
        if type(layer_laterals) == list and len(
                layer_laterals) > block_id:  # If there are some several blocks in the layer we access according to block_id.
            return layer_laterals[block_id]  # We return all lateral associate with the block_id.
        else:
            return layer_laterals  # If there is only 1 lateral connection in the block we return it.
    else:
        return None

class depthwise_separable_conv(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, stride: int = 1, padding: int = 1,
                 bias: bool = False):
        """
        Args:
            channels_in: In channels of the input tensor.
            channels_out: Out channels of the input tensor.
            kernel_size: The kernal size.
            stride: The stride.
            padding: The padding.
            bias: Whether to use bias.
        """
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(channels_in, channels_in, kernel_size=kernel_size, stride=stride, padding=padding,
                                   groups=channels_in,
                                   bias=bias)  # Preserves the number of channels but may downsample by stride.
        self.pointwise = nn.Conv2d(channels_in, channels_out, kernel_size=1,
                                   bias=bias)  # Preserves the inner channels but changes the number of channels.

    def forward(self, x: torch) -> torch:
        """
        Args:
            x: Input tensor to the Conv.

        Returns: Output tensor from the conv.

        """
        out = self.depthwise(x)  # Downsample the tensor.
        out = self.pointwise(out)  # Change the number of channels
        return out


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Module:
    """
    Args:
        in_channels: In channels of the input tensor
        out_channels: Out channels of the output tensor.
        stride: The stride.

    Returns: Module that performs the conv3x3.

    """

    return depthwise_separable_conv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


def conv3x3up(in_channels: int, out_channels: int, size: np.array, upsample=False) -> nn.Module:
    """
    Args:
        in_channels: The number of channels in the input. out_channels < in_channels
        out_channels: The number of channels in the output. out_channels < in_channels
        size: The size to upsample to.
        upsample: Whether to upsample.

    Returns: Module that Upsamples the tensor.

    """
    layer = conv3x3(in_channels, out_channels)  # Changing the number of channels.
    if upsample:
        layer = nn.Sequential(nn.Upsample(size=size, mode='bilinear', align_corners=False),
                              layer)  # Upsample the inner dimensions of the tensor.
    return layer


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Module:
    """
    Args:
        in_channels: In channels of the input tensor
        out_channels: Out channels of the output tensor.
        stride: The stride.

    Returns: 1 by 1 conv.

    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


def num_params(params: list) -> int:
    """
    Args:
        params: The list of parameters.

    Returns: The number of learnable parameters in the list.

    """
    nparams = 0
    for param in params:  # For each parameter in the model we sum its parameters
        cnt = 1
        for p in param.shape:  # The number of params in each weight is the product if its shape.
            cnt = cnt * p
        nparams = nparams + cnt  # Sum for all params.
    return nparams


def create_optimizer_and_sched(opts: argparse, learned_params: list) -> tuple:
    """
    Args:
        opts: The model options.
        learned_params: The learned parameters.

    Returns: Optimizer, scheduler.

    """
    if opts.SGD:
        optimizer = optim.SGD(learned_params, lr=opts.initial_lr, momentum=opts.momentum, weight_decay=opts.wd)
        if opts.cycle_lr:
            scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=opts.base_lr, max_lr=opts.max_lr,
                                                    step_size_up=opts.nbatches_train // 2, step_size_down=None,
                                                    mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle',
                                                    cycle_momentum=True, base_momentum=0.8, max_momentum=0.9,
                                                    last_epoch=-1)
    else:
        optimizer = optim.Adam(learned_params, lr=opts.base_lr, weight_decay=opts.wd)
        if opts.cycle_lr:
            scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=opts.base_lr, max_lr=opts.max_lr,
                                                    step_size_up=opts.nbatches_train // 2, step_size_down=None,
                                                    mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle',
                                                    cycle_momentum=False, last_epoch=-1)

    opts.optimizer = optimizer
    if not opts.cycle_lr:
        lmbda = lambda epoch: opts.learning_rates_mult[epoch]
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)

    return optimizer, scheduler


def Get_learned_params(model,task_id):
    learned_param = []
    learned_param.extend(model.bumodel.parameters())
    learned_param.extend(model.transfer_learning[task_id])
    return learned_param

def preprocess(inputs: torch, device) -> torch:
    # Moves the tensor into the device, usually to the cuda.
    inputs = [inp.to(device) for inp in inputs]
    return inputs
