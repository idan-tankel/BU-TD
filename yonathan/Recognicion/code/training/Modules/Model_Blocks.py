import argparse
from typing import Iterator, Union

import numpy as np
import torch
import torch.nn as nn

from training.Data.Data_params import Flag
from training.Modules.Module_Blocks import conv3x3, conv1x1, conv3x3up, \
    Modulation_and_Lat, Modulation
from training.Utils import Compose_Flag
from training.Utils import flag_to_idx
from training.Utils import get_laterals


# Here we define the Basic BU, TD, BU shared blocks.

class BasicBlockBUShared(nn.Module):
    # Basic block of the shared part between BU1, BU2, contain only the conv layers.
    # Highly Based on the ResNet pytorch's implementation.
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
        self.in_channels = in_channels # The input channel dimension.
        self.out_channels = out_channels  # The output channel dimension.
        self.use_lateral = opts.use_lateral_tdbu # Whether to create the BU -> TD laterals.
        self.conv1 = conv3x3(in_channels, out_channels, stride)  # changes the number of channels and the spatial shape.
        self.conv2 = conv3x3(out_channels, out_channels)  # preserves the tensor shape.
        self.downsample = None
        if stride != 1 or in_channels != out_channels * BasicBlockBUShared.expansion:
            self.downsample = conv1x1(in_channels, out_channels * BasicBlockBUShared.expansion,
                                 stride)  # performs downsmaple on the lateral connection to match the shape after conv1.|
        if self.use_lateral:
            self.nfilters_lat1 = in_channels  # The first lateral connection number of channels.
            self.nfilters_lat2 = out_channels  # The second lateral connection number of channels.
            self.nfilters_lat3 = out_channels  # The third lateral connection number of channels.


class BUInitialBlock(nn.Module):
    def __init__(self, opts: argparse, shared: nn.Module, is_bu2: bool = False):
        """
        Basic BU block.
        Receiving the shared conv layers and initializes other parameters specifically.
        Args:
            opts: The model options.
            shared: The shared part between BU1, BU2.
            is_bu2: Whether the block is part of the BU2 stream.
        """

        super(BUInitialBlock, self).__init__()
        filters = opts.nfilters[0] # The input filters.
        norm_layer = opts.norm_layer  # The norm layer.
        activation_fun = opts.activation_fun  # The activation layer.
        self.ndirections = opts.ndirections # The number of directions.
        # If use laterals, and we are on the second stream, we add lateral connection.
        if opts.use_lateral_tdbu and is_bu2:
            self.bot_lat = Modulation_and_Lat(opts,
                                              shared.nfilters_bot_lat)  # Skip connection from the end of the TD stream.
        self.conv1 = nn.Sequential(shared.conv1, norm_layer(opts, filters),
                                   activation_fun())  # The initial block downsample from RGB.

    def forward(self, inputs: list[torch]) -> torch:
        """
        Args:
            inputs: The images, flags, the lateral connections(if exists).

        Returns: The output if the first block.

        """
        x, flags, laterals_in = inputs  # The input is the image, the flag and the lateral from the previous stream.
        x = self.conv1(x)  # Compute conv1.
        lateral_in = get_laterals(laterals_in, 0, 0)  # The initial lateral connection.
        if lateral_in is not None:
            x = self.bot_lat((x, lateral_in))  # Apply the skip connection.
        return x


class BasicBlockBU(nn.Module):
    # Basic block of the BU1, BU2 streams.
    def __init__(self, opts: argparse, shared: nn.Module, block_inshapes: torch, is_bu2: bool,
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
        nchannels = block_inshapes[0]  # computing the shape for the channel and pixel modulation.
        if self.flag_at is Flag.CL and self.is_bu2:  # If BU2 stream and continual learning mode, create the task embedding.
            shape_spatial = block_inshapes[1:]  # computing the shape for the channel and pixel modulation.
            self.channel_modulation_after_conv1 = Modulation(opts=opts, shape=nchannels, pixel_modulation=False,
                                                             task_embedding=task_embedding)  # channel modulation after conv1.
            self.pixel_modulation_after_conv1 = Modulation(opts=opts, shape=shape_spatial, pixel_modulation=True,
                                                           task_embedding=task_embedding)  # pixel modulation after conv1.
            self.channel_modulation_after_conv2 = Modulation(opts=opts, shape=nchannels, pixel_modulation=False,
                                                             task_embedding=task_embedding)  # channel modulation after conv2.
            self.pixel_modulation_after_conv2 = Modulation(opts=opts, shape=shape_spatial, pixel_modulation=True,
                                                           task_embedding=task_embedding)  # pixel modulation after conv2.
        self.conv1 = nn.Sequential(shared.conv1, opts.norm_layer(opts, nchannels),
                                   opts.activation_fun())  # conv1 Block.
        self.conv2 = nn.Sequential(shared.conv2, opts.norm_layer(opts, nchannels),
                                   opts.activation_fun())  # conv2 Block.
        if shared.downsample is not None:
            # downsample, the skip connection from the beginning of the block if needed.
            downsample = nn.Sequential(shared.downsample, opts.norm_layer(opts, nchannels))
        else:
            downsample = None
        self.downsample = downsample

        if self.use_lateral and is_bu2:
            self.lat1 = Modulation_and_Lat(opts,
                                           shared.nfilters_lat1)  # Lateral connection 1 from the previous stream if exists.
            self.lat2 = Modulation_and_Lat(opts,
                                           shared.nfilters_lat2)  # Lateral connection 2 from the previous stream if exists.
            self.lat3 = Modulation_and_Lat(opts,
                                           shared.nfilters_lat3)  # Lateral connection 3 from the previous stream if exists.

    def forward(self, inputs: tuple[torch]):
        """
        Args:
            inputs: The input, the flag and the lateral connections(if exist).

        Returns: The block output, the lateral connections.

        """
        x, flag, laterals_in = inputs  # The inputs are the x the tensor input, flag, the lateral connection
        # from the last stream.
        if laterals_in is not None:
            lateral1_in, lateral2_in, lateral3_in = laterals_in  # There are 3 connections from the last stream.
        else:
            lateral1_in, lateral2_in, lateral3_in = None, None, None

        laterals_out = []
        if laterals_in is not None:
            x = self.lat1((x, lateral1_in))  # Perform lateral skip connection if needed..
        laterals_out.append(x)
        inp = x  # Save the inp for the skip to the end of the block
        x = self.conv1(x)  # Perform first Conv Block.
        direction_flag, _, _ = Compose_Flag(opts=self.opts, flag=flag)  # Get the direction flag.
        if self.flag_at is Flag.CL and self.is_bu2:  # perform the task embedding if needed.
            x = self.pixel_modulation_after_conv1(x, direction_flag)
            x = self.channel_modulation_after_conv1(x, direction_flag)

        if laterals_in is not None:  # Perform lateral skip connection.
            x = self.lat2((x, lateral2_in))

        laterals_out.append(x)
        x = self.conv2(x)  # Perform second Conv Block.
        if self.flag_at is Flag.CL and self.is_bu2:  # Perform the task embedding if needed.
            x = self.pixel_modulation_after_conv2(x, direction_flag)
            x = self.channel_modulation_after_conv2(x, direction_flag)

        if laterals_in is not None:  # perform lateral skip connection.
            x = self.lat3((x, lateral3_in))

        laterals_out.append(x)
        if self.downsample is not None:  # downsample the input from the beginning of the block to match the expected shape.
            identity = self.downsample(inp)
        else:
            identity = inp

        x = x + identity  # Perform the skip connection.
        return x, laterals_out


class BasicBlockTD(nn.Module):
    # Basic block of the TD stream.
    # The same architecture as in BU just instead of downsample by stride factor we upsample by stride factor.
    expansion = 1

    def __init__(self, opts: argparse, in_channels: int, out_channels: int, stride: int, shape: np.ndarray):
        """
        Args:
            opts: The model options.
            in_channels: In channels from the last block.
            out_channels: Out channels for the last block.
            stride: The stride to upsample according.
            shape: The model input shape, needed for the upsample block.
        """
        super(BasicBlockTD, self).__init__()
        self.ntasks = opts.ntasks
        self.flag_params = [[] for _ in range(self.ntasks)]
        self.use_lateral = opts.use_lateral_butd
        size = tuple(shape[1:])  # The shape to upsample to.
        if self.use_lateral:
            self.lat1 = Modulation_and_Lat(opts, in_channels)
            self.lat2 = Modulation_and_Lat(opts, in_channels)
            self.lat3 = Modulation_and_Lat(opts, out_channels)
        self.conv1 = conv3x3(in_channels, in_channels)  # The first Block conserves the number of channels.
        self.conv1_norm = opts.norm_layer(opts, in_channels)
        self.relu1 = opts.activation_fun()
        self.conv2 = conv3x3up(in_channels, out_channels, size, stride > 1)  # The second Block Upsamples the input.
        self.conv2_norm = opts.norm_layer(opts, out_channels)
        self.relu3 = opts.activation_fun()
        upsample = None
        out_channels = out_channels * BasicBlockTD.expansion
        if stride > 1:
            # upsample the skip connection.
            upsample = nn.Sequential(nn.Upsample(size=size, mode='bilinear', align_corners=False),
                                     conv1x1(in_channels, out_channels, stride=1), opts.norm_layer(opts, out_channels))
        elif in_channels != out_channels:
            upsample = nn.Sequential(conv1x1(in_channels, out_channels, stride=1), opts.norm_layer(opts, out_channels))
        self.upsample = upsample

    def forward(self, inputs: tuple[torch]) -> tuple[torch, list[torch]]:
        """
        Args:
            inputs: The Block input, the lateral connections and the flag.

        Returns: The block output, the lateral connections.

        """
        x, flag, laterals_in = inputs
        if laterals_in is not None:
            lateral1_in, lateral2_in, lateral3_in = laterals_in  # There are 3 lateral connections from the last stream
        else:
            lateral1_in, lateral2_in, lateral3_in = None, None, None
        laterals_out = []
        if laterals_in is not None:
            x = self.lat1((x, lateral1_in))  # perform lateral connection 1.
        laterals_out.append(x)
        inp = x  # store the input from the skip to the end of the block.
        x = self.conv1(x)  # Performs the first conv that preserves the input shape.
        x = self.conv1_norm(x)  # Perform norm layer.
        x = self.relu1(x)  # Perform the activation fun.
        if laterals_in is not None:
            x = self.lat2((x, lateral2_in))  # perform lateral connection 2.
        laterals_out.append(x)
        x = self.conv2(x)  # Performs the second conv that Upsamples the input.
        x = self.conv2_norm(x)  # Perform norm layer.
        if laterals_in is not None:
            x = self.lat3((x, lateral3_in))  # perform lateral connection 3.
        laterals_out.append(x)
        if self.upsample is not None:
            identity = self.upsample(inp)  # Upsample the input if needed,for the skip connection.
        else:
            identity = inp
        x = x + identity  # Performs the skip connection
        x = self.relu3(x)
        return x, laterals_out[::-1]


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
        norm_layer = opts.norm_layer
        activation_fun = opts.activation_fun
        if self.model_flag is Flag.CL:
            # As different embeddings affect each other we separate them into ParameterList and call each of them
            # in different continual trainings.
     #       self.top_td_task_emb = nn.ParameterList(
      #          [nn.Parameter(torch.Tensor(1, self.top_filters // 2)) for _ in range(self.ndirections)])
            # The argument embedding.
            self.top_td_arg_emb = nn.ModuleList(
                [nn.Linear(self.nclasses[i], self.top_filters ) for i in range(self.ntasks)])
            # The linear projection after concatenation of task, arg embedding, and bu1 out.
            self.td_linear_proj = nn.Linear(self.top_filters * 2, self.top_filters)

        if self.model_flag is Flag.TD:
            # The task embedding.
            self.top_td_task_emb = nn.Sequential(nn.Linear(self.ndirections, self.top_filters // 2),
                                                 norm_layer(opts, self.top_filters // 2, dims=1),
                                                 activation_fun())
            # The argument embedding.
            self.top_td_arg_emb = nn.Sequential(nn.Linear(self.nclasses[0], self.top_filters // 2),
                                                norm_layer(opts, self.top_filters // 2, dims=1),
                                                activation_fun())
            # The projection layer.
            self.td_linear_proj = nn.Sequential(nn.Linear(self.top_filters * 2, self.top_filters),
                                                norm_layer(opts, self.top_filters, dims=1), activation_fun())

    def forward(self, inputs: tuple[torch]) -> tuple[torch]:
        """
        Args:
            inputs: The output from BU1 and the flag.

        Returns: The model output.

        """
        bu_out, flag = inputs  # The input to the Block is the bu1 output, flag.
        bu_out = bu_out.squeeze()  # Flatten the bu1 output.

        direction_flag, task_flag, arg_flag = Compose_Flag(self.opts, flag)  # Get the direction, task, argument flags.
        direction_id = flag_to_idx(direction_flag)  # The direction id.
        task_id = flag_to_idx(task_flag)  # The lan index, for Omniglot is means the langauge index.
        B = bu_out.size(0)
        if self.model_flag is Flag.CL:
            # Take the specific direction embedding to avoid forgetting.
          #  top_td_task = self.top_td_task_emb[direction_id].expand(B, -1)
           pass
        # TODO - SUPPORT ALL MODES.
        else:
            top_td_task = self.top_td_task_emb(direction_flag)  # Compute the direction embedding.
            top_td_task = top_td_task.view((-1, self.top_filters // 2))
        top_td_arg = self.top_td_arg_emb[task_id](arg_flag)  # Embed the argument.
        top_td = torch.cat((bu_out,  top_td_arg),
                           dim=1)  # Concatenate the task and argument vectors and bu_out.
        top_td = self.td_linear_proj(top_td).view(
            [-1, self.top_filters, 1, 1])  # Apply the projection layer and resize to match the shape for Upsampling.
        return top_td


def init_module_weights(modules: Iterator[nn.Module], model_flag: Flag) -> None:
    """
    Initializing the module weights according to the original BU-TD paper.
    Args:
        modules: All model's layers
        model_flag: The model flag.

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
            for layer in m.modulations:
                nn.init.xavier_uniform_(layer)
        '''
        elif isinstance(m, InitialEmbeddingBlock):
            if model_flag is Flag.CL:
                for layer in m.top_td_task_emb:
                    nn.init.xavier_uniform_(layer)
            else:
                nn.init.xavier_uniform_(m.top_td_task_emb)
       '''