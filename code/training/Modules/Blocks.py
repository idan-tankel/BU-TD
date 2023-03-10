"""
Here we define the Basic BU, TD, BU shared blocks.
"""
import argparse
from typing import Iterator, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ..Data.Data_params import Flag
from ..Modules.Module_Blocks import conv3x3, conv1x1, conv3x3up, \
    Modulation_and_Lat, conv_with_modulation
from ..Modules.Batch_norm import BatchNorm
from ..Utils import get_laterals
from ..Data.Structs import inputs_to_struct


# Here we define the Basic BU, TD, BU shared blocks.

class BasicBlockBUShared(nn.Module):
    """
    Basic block of the shared part between BU1, BU2, contain only the conv layers.
    Highly Based on the ResNet pytorch's implementation.
    """

    expansion = 1

    def __init__(self, opts: argparse, in_channels: int, out_channels: int, stride: int, padding: int,
                 kernel_size: int):
        """
        Args:
            opts: The model options.
            in_channels: In channel from the previous block.
            out_channels: Out channel of the block for the Next block.
            stride: Stride to perform.
        """
        super(BasicBlockBUShared, self).__init__()
        self.in_channels = in_channels  # The input channel dimension.
        self.out_channels = out_channels  # The output channel dimension.
        self.use_lateral = opts.use_lateral_tdbu  # Whether to create the BU -> TD laterals.
        self.conv1 = conv3x3(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding)  # changes the
        # number of channels and the
        # spatial shape.
        self.conv2 = conv3x3(in_channels=out_channels, out_channels=out_channels,
                             padding=padding,
                             kernel_size=kernel_size)  # preserves the tensor shape.
        self.downsample = None
        if stride != 1 or in_channels != out_channels * BasicBlockBUShared.expansion:
            # performs downsmaple on the lateral connection to match the shape after conv1.
            self.downsample = conv1x1(in_channels=in_channels, out_channels=out_channels * BasicBlockBUShared.expansion,
                                      stride=stride)
        if self.use_lateral:
            self.nfilters_lat1 = in_channels  # The first lateral connection number of channels.
            self.nfilters_lat2 = out_channels  # The second lateral connection number of channels.
            self.nfilters_lat3 = out_channels  # The third lateral connection number of channels.


class BUInitialBlock(nn.Module):
    """
    Initial Block getting the image as an input.
    """

    def __init__(self, opts: argparse, shared: nn.Module, is_bu2: bool = False):
        """

        Receiving the shared conv layers and initializes other parameters specifically.
        Args:
            opts: The model options.
            shared: The shared part between BU1, BU2.
            is_bu2: Whether the block is part of the BU2 stream.
        """

        super(BUInitialBlock, self).__init__()

        self.ndirections: int = opts.data_obj.ndirections  # The number of directions.
        # If we use lateral connections, and we are on the second stream, we add lateral connection.
        self.is_bu2: bool = is_bu2
        self.flag_at: Flag = opts.model_flag
        self.opts: argparse = opts
        self.conv1: nn.Module = shared.conv1  # The initial block downsample from RGB.
        self.bn1 =  BatchNorm(opts, opts.nfilters[0])
        self.relu = nn.ReLU(inplace=True)
        if opts.use_lateral_tdbu and is_bu2:
            self.bot_lat: nn.Module = Modulation_and_Lat(opts=opts,
                             nfilters=opts.nfilters[0])  # Skip connection from the end of the TD stream.

    def forward(self, x: Tensor, samples, laterals_in: Union[list, None]) -> Tensor:
        """
        Args:
            x: The images.
            flags: The samples, needed for BN statistics storing.
            laterals_in: The previous stream laterals(if exist).

        Returns: The output of the first block.

        """
        x = self.conv1(x)  # Compute conv1.
        x = self.bn1(x, samples)
        x = self.relu(x)
        lateral_in = get_laterals(laterals=laterals_in, layer_id=0)  # The initial lateral connection.
        if lateral_in is not None:
            x = self.bot_lat(x=x, samples = samples, lateral=lateral_in)  # Apply the skip connection.
        return x


class BasicBlockBU(nn.Module):
    """
    Basic block of the BU1, BU2 streams.
    """

    def __init__(self, opts: argparse, shared: nn.Module, block_inshapes: Tensor, is_bu2: bool,
                 task_embedding: Union[list, None] = None) -> None:
        """
        Args:
            opts: The model options.
            shared: The shared part between BU1, BU2.
            block_inshapes: The input shape of the block.
            is_bu2: Whether the stream is BU1 or BU2.
            task_embedding: The task embedding list.
        """
        super(BasicBlockBU, self).__init__()
        self.opts = opts
        self.flag_at = opts.model_flag
        self.is_bu2 = is_bu2
        self.ndirections = opts.data_obj.ndirections
        self.use_lateral = shared.use_lateral
        self.relu = nn.ReLU(inplace=True)
        nchannels = block_inshapes[0]  # computing the shape for the channel and column modulation.
        self.weight_modulation = opts.weight_modulation and self.is_bu2

        self.conv1 = conv_with_modulation(opts = opts,conv_layer = shared.conv1,create_modulation = self.is_bu2,
                                          task_embedding=task_embedding)
        self.bn1 = BatchNorm(opts, nchannels)
        self.conv2 = conv_with_modulation(opts = opts,conv_layer = shared.conv2,create_modulation = self.is_bu2,
                                          task_embedding=task_embedding)
        self.bn2 = BatchNorm(opts, nchannels)

        self.downsample = shared.downsample
        if shared.downsample is not None:
            # downsample, the skip connection from the beginning of the block if needed.
            self.conv1x1 =conv_with_modulation(opts = opts,conv_layer = shared.downsample,create_modulation = self.is_bu2,
                                          task_embedding=task_embedding)
            self.last_bn = BatchNorm(opts, nchannels)

        if self.use_lateral and is_bu2:
            # Lateral connection 1 from the previous stream  if exists.
            self.lat1 = Modulation_and_Lat(opts=opts,
                                           nfilters=shared.nfilters_lat1)
            # Lateral connection 2 from the previous stream if exists.
            self.lat2 = Modulation_and_Lat(opts=opts,
                                           nfilters=shared.nfilters_lat2)
            # Lateral connection 3 from the previous stream if exists.
            self.lat3 = Modulation_and_Lat(opts=opts,
                                           nfilters=shared.nfilters_lat3)

    def forward(self, x: Tensor, samples: Tensor, laterals_in: Tensor) -> tuple[Tensor, list[Tensor]]:
        """
        Args:
            x: The model input.
            samples: The samples, needed for BN statistics storing.
            laterals_in: The previous stream laterals(if exists).

        Returns: The block output, the lateral connections.

        """

        if laterals_in is not None:
            lateral1_in, lateral2_in, lateral3_in = laterals_in  # There are 3 connections from the last stream.
        else:
            lateral1_in, lateral2_in, lateral3_in = None, None, None

        laterals_out = []
        if laterals_in is not None:
            x = self.lat1(x=x, samples = samples, lateral=lateral1_in)  # Perform first lateral skip connection.
        laterals_out.append(x)
        inp = x  # Save the inp for the skip to the end of the block

        x = self.conv1(x,samples)
        x = self.bn1(inputs=x, samples = samples)
        x = self.relu(input=x)

        if laterals_in is not None:
            x = self.lat2(x=x, samples = samples, lateral=lateral2_in)  # Perform second lateral skip connection.

        laterals_out.append(x)

        x = self.conv2(x, samples)
        x = self.bn2(inputs=x, samples = samples,)
        x = self.relu(input=x)

        if laterals_in is not None:
            x = self.lat3(x=x, samples = samples, lateral=lateral3_in)  # perform third lateral skip connection.

        laterals_out.append(x)
        if self.downsample is not None:  # downsample the input from the beginning of the
            # block to match the expected shape.
            identity = self.conv1x1(inp, samples)
            identity = self.last_bn(inputs=identity,samples = samples,)
        else:
            identity = inp

        x = x + identity  # Perform the skip connection.
        return x, laterals_out


class InitialEmbeddingBlock(nn.Module):
    """
    The Initial Task embedding at the top of the TD stream.
    """

    def __init__(self, opts: argparse):
        """
        Args:
            opts: The model options.
        """
        super(InitialEmbeddingBlock, self).__init__()
        self.opts = opts
        self.ntasks = opts.data_obj.ntasks
        self.top_filters = opts.nfilters[-1]
        self.model_flag = opts.model_flag
        self.ndirections = opts.data_obj.ndirections
        self.nclasses = opts.data_obj.nclasses
        if self.model_flag is Flag.CL or self.model_flag is Flag.Read_argument:
            # As we use strong task embedding on the BU2 stream we don't need task-vector.
            # The argument embedding.
            self.top_td_arg_emb = nn.ModuleList(
                [nn.Linear(in_features=self.nclasses[i], out_features=self.top_filters) for i in range(self.ntasks)])
            # The linear projection after concatenation of task, arg embedding, and bu1 out.
            self.td_linear_proj = nn.Linear(in_features=self.top_filters * 2, out_features=self.top_filters)

    def forward(self, bu_out: Tensor, samples: inputs_to_struct) -> Tensor:
        """
        Args:
            bu_out: The BU1 output.
            samples: The model samples.

        Returns: The initial block output, match the last BU layer shape.

        """
        task_id = samples.language_index[0]
        arg_flag = samples.char_flags
        # task, argument samples.
        if self.model_flag is not Flag.CL and self.model_flag is not Flag.Read_argument:
            top_td_task = self.top_td_task_emb(input=direction_flag)  # Compute the task embedding.
            top_td_task = top_td_task.view(-1, self.top_filters // 2)  # Reshape.
        else:
            top_td_task = None  # No task embedding is needed.
        top_td_arg = self.top_td_arg_emb[task_id](input=arg_flag)  # Embed the argument.
        if self.model_flag is Flag.CL or self.model_flag is Flag.Read_argument:
            top_td = torch.cat((bu_out, top_td_arg), dim=1)  # Concatenate the argument vectors and bu_out.
        else:
            top_td = torch.cat((bu_out, top_td_task, top_td_arg),
                               dim=1)  # Concatenate the argument vectors, task vector, and bu_out.
        top_td = self.td_linear_proj(input=top_td).view(
            [-1, self.top_filters, 1, 1])  # Apply the projection layer and resize to match the shape for Upsampling.
        return top_td


class BasicBlockTD(nn.Module):
    """
    Basic block of the TD stream.
    The same architecture as in BU just instead of downsample by stride factor we upsample by stride factor.
    """

    expansion = 1

    def __init__(self, opts: argparse, in_channels: int, out_channels: int, stride: int, block_inshape: np.ndarray,
                 index: int, kernel_size: int, padding: int):
        """
        Args:
            opts: The model options.
            in_channels: In channels from the last block.
            out_channels: Out channels for the last block.
            stride: The stride to upsample according.
            block_inshape: The model input shape, needed for the upsample block.
            index: The block index.
        """
        super(BasicBlockTD, self).__init__()
        self.ntasks = opts.data_obj.ntasks  # The number of tasks.
        self.index = index
        self.use_lateral = opts.use_lateral_butd  # Whether to use the laterals from TD to BU.
        self.relu = nn.ReLU(inplace=True)
        size = tuple(block_inshape[1:])  # The shape upsample to.

        if self.use_lateral:
            self.lat1 = Modulation_and_Lat(opts=opts, nfilters=in_channels)  # The first lateral connection.
            self.lat2 = Modulation_and_Lat(opts=opts, nfilters=in_channels)  # The second lateral connection.
            self.lat3 = Modulation_and_Lat(opts=opts, nfilters=out_channels)  # The third lateral connection.
        self.conv1 = conv3x3(in_channels=in_channels, out_channels=in_channels,
                             kernel_size=kernel_size,
                             padding=padding)
        self.bn1 = BatchNorm(opts=opts, num_channels=in_channels)
        self.conv2 = conv3x3up(in_channels=in_channels, out_channels=out_channels, size=size, upsample=stride > 1,
                                kernel_size=kernel_size, padding=padding)
        self.bn2 = BatchNorm(opts=opts, num_channels=out_channels)
        out_channels *= BasicBlockTD.expansion
        self.upsample = None
        if stride > 1:
            # upsample the skip connection.
            self.upsample = nn.Upsample(size=size, mode='bilinear', align_corners=False)
            self.conv1x1 = conv1x1(in_channels=in_channels, out_channels=out_channels)
            self.last_norm = BatchNorm(opts=opts, num_channels=out_channels)

        elif in_channels != out_channels:
            self.upsample = nn.Identity()
            self.conv1x1 = conv1x1(in_channels=in_channels, out_channels=out_channels)
            self.last_norm = BatchNorm(opts=opts, num_channels=out_channels)

    def forward(self, x: Tensor, samples: Tensor, laterals: Tensor) -> tuple[Tensor, list[Tensor]]:
        """
        Args:
            x: The model input.
            samples: The flags, needed for BN statistics storing.
            laterals: The previous stream lateral connections.

        Returns: The block output, the lateral connections.

        """

        if laterals is not None:
            # There are 3 lateral connections from the last stream.
            lateral1_in, lateral2_in, lateral3_in = laterals
        else:
            lateral1_in, lateral2_in, lateral3_in = None, None, None
        laterals_out = []
        if laterals is not None:
            x = self.lat1(x=x, samples=samples, lateral=lateral1_in)  # perform the first lateral connection.
        laterals_out.append(x)
        inp = x  # store the input from the skip to the end of the block.
        x = self.conv1(x)  # Perform the first conv block.
        x = self.bn1(inputs=x, samples=samples)
        x = self.relu(input=x)
        if laterals is not None:
            x = self.lat2(x=x, samples=samples, lateral=lateral2_in)  # perform the second lateral connection.
        laterals_out.append(x)
        x = self.conv2(x)  # Perform the first conv block.
        x = self.bn2(inputs=x, samples=samples)
        if laterals is not None:
            x = self.lat3(x=x, samples=samples, lateral=lateral3_in)  # perform the third lateral connection.
        laterals_out.append(x)
        if self.upsample is not None:
            identity = self.upsample(input=inp)  # Upsample the input if needed,for the skip connection.
            identity = self.conv1x1(input=identity)
            identity = self.last_norm(inputs=identity, samples=samples)
        else:
            identity = inp
        x = x + identity  # Performs the skip connection
        x = self.relu(input=x)
        return x, laterals_out[::-1]


def init_module_weights(modules: Iterator[nn.Module]) -> None:
    """
    Initializing the module weights according to the original BU-TD paper.
    Args:
        modules: All model's layers

    """
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, Modulation_and_Lat):
            nn.init.xavier_uniform_(m.side)


        elif isinstance(m, conv_with_modulation):
            if hasattr(m, 'modulation'):
                for param in m.modulation:
                    nn.init.kaiming_normal_(param, nonlinearity="relu")
