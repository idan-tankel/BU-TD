import argparse
from typing import Iterator, Union

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

from training.Data.Data_params import Flag
from training.Modules.Module_Blocks import conv3x3, conv1x1, conv3x3up, \
    Modulation_and_Lat, Modulation
from training.Utils import Compose_Flag
from training.Utils import flag_to_idx
from training.Utils import get_laterals


# Here we define the Basic BU, TD, BU shared blocks.

class BasicBlockBUShared(nn.Module):
    """
    Basic block of the shared part between BU1, BU2, contain only the conv layers.
    Highly Based on the ResNet pytorch's implementation.
    """

    expansion = 1

    def __init__(self, opts: argparse, in_channels: int, out_channels: int, stride: int):
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
        self.conv1 = conv3x3(in_channels=in_channels, out_channels=out_channels,
                             stride=stride)  # changes the number of channels and the spatial shape.
        self.conv2 = conv3x3(in_channels=out_channels, out_channels=out_channels)  # preserves the tensor shape.
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
        norm_layer = opts.norm_layer  # The norm layer.
        activation_fun = opts.activation_fun  # The activation layer.
        self.ndirections = opts.ndirections  # The number of directions.
        # If we use lateral connections, and we are on the second stream, we add lateral connection.
        self.is_bu2 = is_bu2
        self.flag_at = opts.model_flag
        self.opts = opts
        self.conv1 = nn.Sequential(shared.conv1, norm_layer(opts, opts.nfilters[0]),
                                   activation_fun())  # The initial block downsample from RGB.
        if opts.use_lateral_tdbu and is_bu2:
            self.bot_lat = Modulation_and_Lat(opts=opts,
                                              nfilters=opts.nfilters[
                                                  0])  # Skip connection from the end of the TD stream.

    def forward(self, x: Tensor, flags, laterals_in: Union[list, None]) -> Tensor:
        """
        Args:
            x: The images.
            flags: The flags, needed for BN statistics storing.
            laterals_in: The previous stream laterals(if exist).

        Returns: The output of the first block.

        """
        x = self.conv1[0](x)  # Compute conv1.
        x = self.conv1[1](x, flags)
        x = self.conv1[2](x)
        lateral_in = get_laterals(laterals=laterals_in, layer_id=0, block_id=0)  # The initial lateral connection.
        if lateral_in is not None:
            x = self.bot_lat(x=x, flags=flags, lateral=lateral_in)  # Apply the skip connection.
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
        self.ndirections = opts.ndirections
        self.use_lateral = shared.use_lateral
        nchannels = block_inshapes[0]  # computing the shape for the channel and column modulation.
        norm_layer = opts.norm_layer
        if self.flag_at is Flag.CL and self.is_bu2:  # If BU2 stream and continual learning mode,
            # we create the task embedding.
            shape_spatial = block_inshapes[1:]  # computing the shape for the channel and column modulation.
            # channel modulation after conv1.
            self.channel_modulation_after_conv1 = Modulation(opts=opts, shape=nchannels, column_modulation=False,
                                                             task_embedding=task_embedding)
            # column modulation after conv1.
            self.column_modulation_after_conv1 = Modulation(opts=opts, shape=shape_spatial, column_modulation=True,
                                                            task_embedding=task_embedding)
            # channel modulation after conv2.
            self.channel_modulation_after_conv2 = Modulation(opts=opts, shape=nchannels, column_modulation=False,
                                                             task_embedding=task_embedding)
            # column modulation after conv2.
            self.column_modulation_after_conv2 = Modulation(opts=opts, shape=shape_spatial, column_modulation=True,
                                                            task_embedding=task_embedding)
        self.conv_block1 = nn.Sequential(shared.conv1, norm_layer(opts, nchannels),
                                         opts.activation_fun())  # conv1 Block.
        self.conv_block2 = nn.Sequential(shared.conv2, norm_layer(opts, nchannels),
                                         opts.activation_fun())  # conv2 Block.
        if shared.downsample is not None:
            # downsample, the skip connection from the beginning of the block if needed.
            self.downsample = nn.Sequential(shared.downsample, norm_layer(opts, nchannels))
        else:
            self.downsample = None

        if self.use_lateral and is_bu2:
            self.lat1 = Modulation_and_Lat(opts=opts,
                                           nfilters=shared.nfilters_lat1)  # Lateral connection 1 from the previous stream
            # if exists.
            self.lat2 = Modulation_and_Lat(opts=opts,
                                           nfilters=shared.nfilters_lat2)  # Lateral connection 2 from the previous stream
            # if exists.
            self.lat3 = Modulation_and_Lat(opts=opts,
                                           nfilters=shared.nfilters_lat3)  # Lateral connection 3 from the previous stream
            # if exists.

    def forward(self, x: Tensor, flags: Tensor, laterals_in: Tensor):
        """
        Args:
            x: The model input.
            flags: The flags, needed for BN statistics storing.
            laterals_in: The previous stream laterals(if exists).

        Returns: The block output, the lateral connections.

        """

        if laterals_in is not None:
            lateral1_in, lateral2_in, lateral3_in = laterals_in  # There are 3 connections from the last stream.
        else:
            lateral1_in, lateral2_in, lateral3_in = None, None, None

        laterals_out = []
        if laterals_in is not None:
            x = self.lat1(x=x, flags=flags, lateral=lateral1_in)  # Perform first lateral skip connection.
        laterals_out.append(x)
        inp = x  # Save the inp for the skip to the end of the block
        x = self.conv_block1[0](x=x)  # Perform first Conv Block.
        x = self.conv_block1[1](inputs=x, flags=flags)
        x = self.conv_block1[2](input=x)
        direction_flag, _, _ = Compose_Flag(opts=self.opts, flags=flags)  # Get the direction flag.
        if self.flag_at is Flag.CL and self.is_bu2:  # perform the first task embedding if needed.
            x = self.column_modulation_after_conv1(x=x, flags=direction_flag)
            x = self.channel_modulation_after_conv1(x=x, flags=direction_flag)

        if laterals_in is not None:
            x = self.lat2(x=x, flags=flags, lateral=lateral2_in)  # Perform second lateral skip connection.

        laterals_out.append(x)
        x = self.conv_block2[0](x=x)  # Perform second Conv Block.
        x = self.conv_block2[1](inputs=x, flags=flags)
        x = self.conv_block2[2](input=x)
        if self.flag_at is Flag.CL and self.is_bu2:  # Perform the second task embedding if needed.
            x = self.column_modulation_after_conv2(x=x, flags=direction_flag)
            x = self.channel_modulation_after_conv2(x=x, flags=direction_flag)

        if laterals_in is not None:
            x = self.lat3(x=x, flags=flags, lateral=lateral3_in)  # perform third lateral skip connection.

        laterals_out.append(x)
        if self.downsample is not None:  # downsample the input from the beginning of the
            # block to match the expected shape.
            identity = self.downsample[0](input=inp)
            identity = self.downsample[1](inputs=identity, flags=flags)
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
        self.ntasks = opts.ntasks
        self.top_filters = opts.nfilters[-1]
        self.model_flag = opts.model_flag
        self.ndirections = opts.ndirections
        self.nclasses = opts.nclasses
        if self.model_flag is Flag.CL:
            # As we use strong task embedding on the BU2 stream we don't need task-vector.
            # The argument embedding.
            self.top_td_arg_emb = nn.ModuleList(
                [nn.Linear(in_features=self.nclasses[i], out_features=self.top_filters) for i in range(self.ntasks)])
            # The linear projection after concatenation of task, arg embedding, and bu1 out.
            self.td_linear_proj = nn.Linear(in_features=self.top_filters * 2, out_features=self.top_filters)

        if self.model_flag is Flag.TD:
            # The task embedding.
            self.top_td_task_emb = nn.Linear(in_features=self.ndirections, out_features=self.top_filters // 2)

            # The argument embedding.
            self.top_td_arg_emb = nn.Linear(in_features=self.nclasses[0], out_features=self.top_filters // 2)

            # The projection layer.
            self.td_linear_proj = nn.Linear(in_features=self.top_filters * 2, out_features=self.top_filters)

    def forward(self, bu_out: Tensor, flags: Tensor) -> tuple[Tensor]:
        """
        Args:
            bu_out: The BU1 output.
            flags: The model flags.

        Returns: The initial block output, match the last BU layer shape.

        """
        direction_flag, task_flag, arg_flag = Compose_Flag(opts=self.opts,
                                                           flags=flags)  # Get the direction, task, argument flags.
        task_id = flag_to_idx(flags=task_flag)  # The lan index, for Omniglot it means the langauge index.
        if self.model_flag is not Flag.CL:
            top_td_task = self.top_td_task_emb(input=direction_flag)  # Compute the direction embedding.
            top_td_task = top_td_task.view(-1, self.top_filters // 2)  # Reshape.
        else:
            top_td_task = None  # No task embedding is needed.
        top_td_arg = self.top_td_arg_emb[task_id](input=arg_flag)  # Embed the argument.
        if self.model_flag is Flag.CL:
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

    def __init__(self, opts: argparse, in_channels: int, out_channels: int, stride: int, block_inshape: np.ndarray, index: int):
        """
        Args:
            opts: The model options.
            in_channels: In channels from the last block.
            out_channels: Out channels for the last block.
            stride: The stride to upsample according.
            block_inshape: The model input shape, needed for the upsample block.
        """
        super(BasicBlockTD, self).__init__()
        self.ntasks = opts.ntasks  # The number of tasks.
        self.index = index
        self.use_lateral = opts.use_lateral_butd  # Whether to use the laterals from TD to BU.
        size = tuple(block_inshape[1:])  # The shape upsample to.
        norm_layer = opts.norm_layer
        activation_fun = opts.activation_fun
        if self.use_lateral:
            self.lat1 = Modulation_and_Lat(opts=opts, nfilters=in_channels)  # The first lateral connection.
            self.lat2 = Modulation_and_Lat(opts=opts, nfilters=in_channels)  # The second lateral connection.
            self.lat3 = Modulation_and_Lat(opts=opts, nfilters=out_channels)  # The third lateral connection.
        self.conv_block1 = nn.Sequential(conv3x3(in_channels=in_channels, out_channels=in_channels),
                                         norm_layer(opts=opts, num_channels=in_channels),
                                         activation_fun())
        self.conv_block2 = nn.Sequential(
            conv3x3up(in_channels=in_channels, out_channels=out_channels, size=size, upsample=stride > 1),
            norm_layer(opts=opts, num_channels=out_channels))
        self.relu3 = activation_fun()  # The third AF.
        out_channels *= BasicBlockTD.expansion
        self.upsample = None
        if stride > 1:
            # upsample the skip connection.
            self.upsample = nn.Sequential(nn.Upsample(size=size, mode='bilinear', align_corners=False),
                                          conv1x1(in_channels=in_channels, out_channels=out_channels),
                                          norm_layer(opts=opts, num_channels=out_channels))
        elif in_channels != out_channels:
            self.upsample = nn.Sequential(nn.Identity(), conv1x1(in_channels=in_channels, out_channels=out_channels),
                                          norm_layer(opts=opts, num_channels=out_channels))

    def forward(self, x: Tensor, flags, laterals: Tensor) -> tuple[Tensor, list[Tensor]]:
        """
        Args:
            x: The model input.
            flags: The flags, needed for BN statistics storing.
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
            x = self.lat1(x=x, flags=flags, lateral=lateral1_in)  # perform the first lateral connection.
        laterals_out.append(x)
        inp = x  # store the input from the skip to the end of the block.
        x = self.conv_block1[0](x=x)  # Perform the first conv block.
        x = self.conv_block1[1](inputs=x, flags=flags)
        x = self.conv_block1[2](input=x)
        if laterals is not None:
            x = self.lat2(x=x, flags=flags, lateral=lateral2_in)  # perform the second lateral connection.
        laterals_out.append(x)
        x = self.conv_block2[0](x)  # Perform the first conv block.
        x = self.conv_block2[1](inputs=x, flags=flags)
        if laterals is not None:
            x = self.lat3(x=x, flags=flags, lateral=lateral3_in)  # perform the third lateral connection.
        laterals_out.append(x)
        if self.upsample is not None:
            identity = self.upsample[0](input=inp)  # Upsample the input if needed,for the skip connection.
            identity = self.upsample[1](input=identity)
            identity = self.upsample[2](inputs=identity, flags=flags)
        else:
            identity = inp
        x = x + identity  # Performs the skip connection
        x = self.relu3(input=x)
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

        elif isinstance(m, Modulation):
            for param in m.modulations:
                nn.init.xavier_uniform_(param)
