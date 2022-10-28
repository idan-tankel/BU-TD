import argparse

import numpy as np
import torch
import torch.nn as nn

from supp.Dataset_and_model_type_specification import Flag
from supp.Module_Blocks import conv3x3, conv1x1, conv3x3up
from supp.utils import get_laterals, flag_to_idx
from supp.Module_Blocks import SideAndComb, Modulation


# Here we define the Basic BU, TD, BU shared blocks.

class BasicBlockBUShared(nn.Module):
    # Basic block of the shared part between BU1, BU2, especially the conv layers.
    # Based at most on the ResNet pytorch's implementation.
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
        self.in_channels = in_channels
        self.out_channels = out_channels  # storing variables.
        self.use_lateral = opts.use_lateral_tdbu
        self.conv1 = conv3x3(in_channels, out_channels, stride)  # changes the number of channels and the spatial shape.
        self.conv2 = conv3x3(out_channels, out_channels)  # preserves the tensor shape
        downsample = None
        if stride != 1 or in_channels != out_channels * BasicBlockBUShared.expansion:
            downsample = conv1x1(in_channels, out_channels * BasicBlockBUShared.expansion,
                                 stride)  # performs downsmaple on the lateral connection to match the shape after conv1.
        self.downsample = downsample
        if self.use_lateral:
            self.nfilters_lat1 = in_channels  # The lateral connection from the previous stream.
            self.nfilters_lat2 = out_channels  # The lateral connection from the previous stream.
            self.nfilters_lat3 = out_channels  # The lateral connection from the previous stream.


class BUInitialBlock(nn.Module):
    def __init__(self, opts: argparse, shared: nn.Module):
        """
        Basic BU block.
        Receiving the shared convs and initializes other parameters specifically.
        Args:
            opts: The shared part between BU1, BU2.
            shared: The shared part between BU1, BU2.
        """

        super(BUInitialBlock, self).__init__()
        filters = opts.nfilters[0]
        norm_layer = opts.norm_layer  # The norm layer.
        activation_fun = opts.activation_fun  # The activation layer.
        use_lateral = opts.use_lateral_tdbu
        if use_lateral:
            self.bot_lat = SideAndComb(opts, shared.nfilters_bot_lat)  # Skip connection from the TD initial embedding.
        self.conv1 = nn.Sequential(shared.conv1, norm_layer(opts, filters),
                                   activation_fun())  # The initial block downsampling from RGB.

    def forward(self, inputs: list[torch]) -> torch:
        """
        Args:
            inputs: The images, flags, the lateral connections(if exists).

        Returns: The output if the first block.

        """
        x, flags, laterals_in = inputs  # The input is the image, the flag and the lateral from the previous stream.
        x = self.conv1(x)  # Compute conv1.
        lateral_in = get_laterals(laterals_in, 0, 0)  # The initial lateral connection.
        if lateral_in is not None:  # Apply the skip connection.
            x = self.bot_lat((x, lateral_in))  # Compute the skip connection.
        return x


class BasicBlockBU(nn.Module):
    # Basic block of the BU1,BU2 streams.
    def __init__(self, opts: argparse, shared: nn.Module, block_inshapes: torch, is_bu2: bool,
                 task_embedding: list = []) -> None:
        """
        Args:
            opts: The model options.
            shared: The shared part between BU1, BU2.
            block_inshapes: The input shape of the block.
            is_bu2: Whether the stream is BU1 or BU2.
            task_embedding: The task embedding list.
        """
        super(BasicBlockBU, self).__init__()
        self.flag_at = opts.model_flag
        self.is_bu2 = is_bu2
        self.ndirections = opts.ndirections
        self.use_lateral = shared.use_lateral
        nchannels = block_inshapes[0]  # computing the shape for the channel and pixel modulation.
        if self.flag_at is Flag.CL and self.is_bu2:  # If BU2 stream create the task embedding.
            shape_spatial = block_inshapes[1:]  # computing the shape for the channel and pixel modulation.
            self.channel_modulation_after_conv1 = Modulation(opts=opts, shape=nchannels, pixel_modulation=False,
                                                             task_embedding=task_embedding)  # channel modulation after conv1.
            self.pixel_modulation_after_conv1 = Modulation(opts=opts, shape=shape_spatial, pixel_modulation=True,
                                                           task_embedding=task_embedding)  # pixel modulation after conv1.
            self.channel_modulation_after_conv2 = Modulation(opts=opts, shape=nchannels, pixel_modulation=False,
                                                             task_embedding=task_embedding)  # channel modulation after conv2.
            self.pixel_modulation_after_conv2 = Modulation(opts=opts, shape=shape_spatial, pixel_modulation=True,
                                                           task_embedding=task_embedding)  # pixel modulation after conv2.
        self.conv1 = nn.Sequential(shared.conv1, opts.norm_layer(opts, nchannels), opts.activation_fun())  # conv1
        self.conv2 = nn.Sequential(shared.conv2, opts.norm_layer(opts, nchannels), opts.activation_fun())  # conv2
        if shared.downsample is not None:
            downsample = nn.Sequential(shared.downsample, opts.norm_layer(opts,
                                                                          nchannels))  # downsample, the skip connection from the beginning of the block if needed.
        else:
            downsample = None
        self.downsample = downsample

        if self.use_lateral:
            self.lat1 = SideAndComb(opts,
                                    shared.nfilters_lat1)  # Lateral connection 1 from the previous stream if exists.
            self.lat2 = SideAndComb(opts,
                                    shared.nfilters_lat2)  # Lateral connection 1 from the previous stream if exists.
            self.lat3 = SideAndComb(opts,
                                    shared.nfilters_lat3)  # Lateral connection 1 from the previous stream if exists.

    def forward(self, inputs: tuple[torch]):
        """
        Args:
            inputs: The input, the flag and the lateral connections if exist.

        Returns: The block output, the lateral connections.

        """
        x, flag, laterals_in = inputs  # The inputs are the x(from the previous block or an image) , flag , the lateral connection from the last stream (if exists).
        if laterals_in is not None:
            lateral1_in, lateral2_in, lateral3_in = laterals_in  # There are 3 connections from the last stream.
        else:
            lateral1_in, lateral2_in, lateral3_in = None, None, None

        laterals_out = []
        if laterals_in is not None:
            x = self.lat1((x, lateral1_in))  # perform lateral skip connection.
        laterals_out.append(x)
        inp = x  # save the inp for the skip to the end of the block
        x = self.conv1(x)  # perform conv
        if self.flag_at is Flag.CL and self.is_bu2:  # perform the task embedding if needed.
            direction_flag = flag[:, :self.ndirections]
            x = self.pixel_modulation_after_conv1(x, direction_flag)
            x = self.channel_modulation_after_conv1(x, direction_flag)

        if laterals_in is not None:  # perform lateral skip connection.
            x = self.lat2((x, lateral2_in))

        laterals_out.append(x)
        x = self.conv2(x)  # perform conv
        if self.flag_at is Flag.CL and self.is_bu2:  # perform the task embedding if needed.
            direction_flag = flag[:, :self.ndirections]
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


class InitialTaskEmbedding(nn.Module):
    """
    The Initial Task embedding at the top of the TD stream.
    """

    def __init__(self, opts: argparse, task_embedding: list):
        """
        Args:
            opts: The model options.
            task_embedding: List containing the task embedding.
        """
        super(InitialTaskEmbedding, self).__init__()
        self.ntasks = opts.ntasks
        self.top_filters = opts.nfilters[-1]
        self.model_flag = opts.model_flag
        self.ndirections = opts.ndirections
        self.nclasses = opts.nclasses
        self.ds_type = opts.ds_type
        norm_layer = opts.norm_layer
        activation_fun = opts.activation_fun
        if self.model_flag is Flag.CL:
            # As different embeddings affect each other we separate them into ModuleList and call each of them in different continual trainings.
            self.h_flag_arg_td = []
            self.td_linear_proj = nn.Sequential(nn.Linear(self.top_filters * 2, self.top_filters),
                                                # The linear projection after concatenation with task and arg embedding.
                                                norm_layer(opts, self.top_filters, dims=1), activation_fun())

            self.top_td_task_emb = nn.ModuleList(
                [nn.Sequential(nn.Linear(1, self.top_filters // 2),  # The task embedding.
                               norm_layer(opts, self.top_filters // 2, dims=1),
                               activation_fun()) for _ in range(self.ndirections)])
            Top_td_task_emb_params = [list(layer.parameters()) for layer in self.top_td_task_emb]
            [task_embedding[i].extend(Top_td_task_emb_params[i]) for i in range(self.ndirections)]
            # The argument embedding.
            self.top_td_arg_emb = nn.ModuleList([nn.Sequential(nn.Linear(self.nclasses[i], self.top_filters // 2),
                                                               norm_layer(opts, self.top_filters // 2, dims=1),
                                                               activation_fun()) for i in range(self.ntasks)])

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
        (bu_out, flag) = inputs
        direction_flag = flag[:, :self.ndirections]  # The direction vector.
        task_flag = flag[:, self.ndirections:self.ndirections + self.ntasks]  # The task vector.
        arg_flag = flag[:, self.ndirections + self.ntasks:]  # The argument vector.
        direction_id = flag_to_idx(direction_flag)  # The direction id.
        task_id = flag_to_idx(task_flag)  # The lan id.
        ones = direction_flag[:, direction_id].view([-1, 1])  # The input to the task embedding.

        if self.model_flag is Flag.CL:
            top_td_task = self.top_td_task_emb[direction_id](
                ones)  # Take the specific task embedding to avoid forgetting.
        else:
            top_td_task = self.top_td_task_emb(direction_flag)
        top_td_task = top_td_task.view((-1, self.top_filters // 2, 1, 1))
        top_td_arg = self.top_td_arg_emb[task_id](arg_flag)  # Embed the argument.
        top_td_arg = top_td_arg.view((-1, self.top_filters // 2, 1, 1))
        top_td = torch.cat((top_td_task, top_td_arg), dim=1)  # Concatenate the flags
        h_side_top_td = bu_out
        top_td = torch.cat((h_side_top_td, top_td), dim=1)  # Concatenate with BU1 output.
        top_td = torch.flatten(top_td, 1)
        top_td = self.td_linear_proj(top_td)  # Apply the projection layer.
        top_td = top_td.view((-1, self.top_filters, 1, 1))
        return top_td


class BasicBlockTD(nn.Module):
    # Basic block of the TD stream.
    # The same architecture as in BU just instead of downsampling by stride factor we upsample by stride factor.
    expansion = 1

    def __init__(self, opts: argparse, in_channels: int, out_channels: int, stride: int, shape: np.ndarray):
        """
        Args:
            opts: The model options.
            in_channels: In channels from the last block.
            out_channels: Out channels for the last block.
            stride: The stride to upsample according.
            shape: The model inshapes, needed for the upsampling block.
        """
        super(BasicBlockTD, self).__init__()
        self.ntasks = opts.ntasks
        self.flag_params = [[] for _ in range(self.ntasks)]
        self.use_lateral = opts.use_lateral_butd
        size = tuple(shape[1:])
        if self.use_lateral:
            self.lat1 = SideAndComb(opts, in_channels)
            self.lat2 = SideAndComb(opts, in_channels)
            self.lat3 = SideAndComb(opts, out_channels)
        self.conv1 = conv3x3(in_channels, in_channels)
        self.conv1_norm = opts.norm_layer(opts, in_channels)
        self.relu1 = opts.activation_fun()
        self.conv2 = conv3x3up(in_channels, out_channels, size, stride > 1)
        self.conv2_norm = opts.norm_layer(opts, out_channels)
        self.relu3 = opts.activation_fun()
        upsample = None
        out_channels = out_channels * BasicBlockTD.expansion
        if stride > 1:
            upsample = nn.Sequential(nn.Upsample(size=size, mode='bilinear', align_corners=False),
                                     # Updample the skip connection.
                                     conv1x1(in_channels, out_channels, stride=1), opts.norm_layer(opts, out_channels))
        elif in_channels != out_channels:
            upsample = nn.Sequential(conv1x1(in_channels, out_channels, stride=1), opts.norm_layer(opts, out_channels))
        self.upsample = upsample

    def forward(self, inputs):
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
            x = self.lat1((x, lateral1_in))  # perform lateral connection1
        laterals_out.append(x)
        inp = x  # store the input from the skip to the end of the block.
        x = self.conv1(x)  # Performs conv that preserves the input shape
        x = self.conv1_norm(x)  # Perform norm layer.
        x = self.relu1(x)  # Perform the activation fun
        if laterals_in is not None:
            x = self.lat2((x, lateral2_in))  # perform lateral connection2
        laterals_out.append(x)
        x = self.conv2(x)  # Performs conv that upsamples the input
        x = self.conv2_norm(x)  # Perform norm layer.
        if laterals_in is not None:
            x = self.lat3((x, lateral3_in))  # perform lateral connection3
        laterals_out.append(x)
        if self.upsample is not None:
            identity = self.upsample(inp)  # Upsample the input if needed,for the skip connection.
        else:
            identity = inp
        x = x + identity  # Performs the skip connection
        x = self.relu3(x)
        return x, laterals_out[::-1]
