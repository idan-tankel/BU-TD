import argparse

import numpy as np
import torch
import torch.nn as nn

from supp.Dataset_and_model_type_specification import Flag
from supp.general_functions import conv3x3, conv1x1, conv3x3up, get_laterals, flag_to_task


class ChannelModulation(nn.Module):
    # The layer performs the channel modulation on the lateral connection
    def __init__(self, nchannels: int):
        """
        Args:
            nchannels: The number of channels to perform channel-modulation on.
        """
        super(ChannelModulation, self).__init__()
        self.nchannels = nchannels
        shape = [self.nchannels, 1, 1]
        self.weights = nn.Parameter(
            torch.Tensor(*shape))  # creates the learnable parameter of shape [nchannels,1,1] according to nchannels.

    def forward(self, inputs: torch) -> torch:
        """
        Perform the channel modulation.
        Args:
            inputs: Tensor of shape [N,C,H,W].

        Returns: Tensor of shape [N,C,H,W].

        """
        return inputs * self.weights  # performs the channel-modulation.


def init_module_weights(modules: nn.Module) -> None:
    """
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
        elif isinstance(m, ChannelModulation):
            nn.init.xavier_uniform_(m.weights)  # init your weights here...


class SideAndComb(nn.Module):
    # performs the lateral connection BU1 -> TD or TD -> BU2
    def __init__(self, opts: argparse, filters: int):
        """
        Args:
            opts: The model options.
            filters: The number of filters.
        """
        super(SideAndComb, self).__init__()
        self.side = ChannelModulation(filters)  # channel-modulation layer.
        # batch norm after the channel-modulation of the lateral.
        self.norm = opts.norm_layer(opts, filters)
        self.filters = filters
        self.relu1 = opts.activation_fun()  # activation_fun after the batch_norm layer
        self.relu2 = opts.activation_fun()  # activation_fun after the skip connection

    def forward(self, inputs: torch) -> torch:
        """
        Args:
            inputs: Two tensors, the first is the input and the second is the lateral connection.

        Returns: The output after the lateral connection.

        """
        x, lateral = inputs  # input, lateral connection.
        side_val = self.side(lateral)  # channel-modulation(CM)
        side_val = self.norm(side_val)  # batch_norm after the CM
        side_val = self.relu1(side_val)  # activation_fun after the batch_norm
        x = x + side_val  # the lateral skip connection
        x = self.relu2(x)  # activation_fun after the skip connection
        return x


class SideAndCombSharedBase:
    # Class saving the needed number of filters in the lateral connection for BU1, BU2.
    def __init__(self, filters):
        """
        Args:
            filters: The number of filters.
        """
        self.filters = filters


class Modulation(nn.Module):  # Modulation layer.
    def __init__(self, opts, shape: list, pixel_modulation: bool, task_embedding: list):
        """
        Channel & pixel modulation layer.
        Args:
            opts: The model options.
            shape: The shape to create the model according to.
            pixel_modulation: Whether to create pixel/channel modulation.
            task_embedding: The task embedding.
        """
        super(Modulation, self).__init__()
        self.inshapes = shape
        self.pixel_modulation = pixel_modulation
        self.modulation = []
        if self.pixel_modulation:
            # If pixel modulation matches the inner spatial of the input
            self.size = [-1, 1, *shape]
        else:
            # If channel modulation matches the number of channels
            self.size = [-1, shape, 1, 1]
        inshapes = np.prod(shape)
        for i in range(opts.ndirections):  # allocating for every task its task embedding
            layer = nn.Linear(1, inshapes, bias=True)
            task_embedding[i].extend(list(layer.parameters()))
            # TODO since thins is a PYTHON GENERATOR object, it's a bit uneficient to extend it every time.
            # The other way to do that is to iterate over it (by, for example, using for item in layer.parameters()) or to use the next method (x=layer.parameters(); x.next()).
            self.modulation.append(layer)
        self.modulation = nn.ModuleList(self.modulation)

    def forward(self, inputs: torch, flag: torch) -> torch:
        """
        perform the channel/pixel modulation.
        Args:
            inputs: Torch of shape [B,C,H,W] to modulate.
            flag: torch of shape [B,S].

        Returns: Torch of shape [B,C,H,W], the modulated tensor.

        """
        task_idx = flag_to_task(flag)
        flag_task = flag[:, task_idx].view(-1, 1)
        task_emb = self.modulation[task_idx](flag_task).view(
            self.size)  # computed the task embedding according to the task_idx and changes the shape according to pixel_modulation. # compute the task embedding according to the task_idx.
        inputs = inputs * (1 - task_emb)  # perform the modulation.
        return inputs


class BasicBlockBUShared(nn.Module):
    # Basic block of the shared part between BU1,BU2.
    # Based at most on the ResNet pytorch's implementation.
    expansion = 1

    def __init__(self, opts: argparse, in_channels: int, out_channels: int, stride: int, idx: int = 0):
        """
        Args:
            opts: The model options.
            in_channels: In channel from the previous block.
            out_channels: Out channel of the block for the Next block.
            stride: Stride to perform.
            idx: The block index.
        """
        super(BasicBlockBUShared, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels  # storing variables.
        self.stride = stride
        self.use_lateral = opts.use_lateral_tdbu
        self.index = idx
        # changes the number of channels and the spatial shape.
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        # preserves the tensor shape
        self.conv2 = conv3x3(out_channels, out_channels)
        downsample = None
        if stride != 1 or in_channels != out_channels * BasicBlockBUShared.expansion:
            downsample = conv1x1(in_channels, out_channels * BasicBlockBUShared.expansion,
                                 stride)  # performs downsmaple on the lateral connection to match the shape after conv1.
        self.downsample = downsample
        if self.use_lateral:
            # The lateral connection from the previous stream.
            self.lat1 = SideAndCombSharedBase(filters=in_channels)
            # The lateral connection from the previous stream.
            self.lat2 = SideAndCombSharedBase(filters=out_channels)
            # The lateral connection from the previous stream.
            self.lat3 = SideAndCombSharedBase(filters=out_channels)


class BUInitialBlock(nn.Module):
    def __init__(self, opts: argparse, shared: nn.Module):
        """
        Basic BU block.
        Args:
            opts: The shared part between BU1, BU2.
            shared: The shared part between BU1, BU2.
        """

        super(BUInitialBlock, self).__init__()
        self.filters = opts.nfilters[0]
        self.norm_layer = opts.norm_layer
        self.activation_fun = opts.activation_fun
        self.ntasks = opts.ntasks
        self.use_lateral = opts.use_lateral_tdbu
        # The initial block downsampling from RGB.
        self.conv1 = nn.Sequential(shared.conv1, opts.norm_layer(
            num_channels=opts.ntasks, self.filters), self.activation_fun())
        if self.use_lateral:
            # Skip connection from the TD initial embedding.
            self.bot_lat = SideAndComb(opts, shared.bot_lat.filters)

    def forward(self, inputs: list[torch]) -> torch:
        """
        Args:
            inputs: The images, flags, the lateral connections(if exists).

        Returns: The output if the first block.

        """
        x, flags, laterals_in = inputs  # The input is the image, the flag and the lateral from the previous stream.
        x = self.conv1(x)  # Compute conv1.
        # The initial lateral connection.
        lateral_in = get_laterals(laterals_in, 0, None)
        if lateral_in is not None:
            x = self.bot_lat((x, lateral_in))  # Compute the skip connection.
        return x


class BasicBlockBU(nn.Module):
    # Basic block of the BU1,BU2 streams.
    def __init__(self, opts: argparse.ArgumentParser, shared: nn.Module, block_inshapes: torch.Tensor, is_bu2: bool, task_embedding=None) -> None:
        """
        Args:
            opts: The model options.
            shared: The shared part between BU1, BU2.
            block_inshapes: The input shape of the block.
            is_bu2: is the stream BU1 or BU2.
            task_embedding(list): The task embedding. For later use under `Modulation` class representing channel moduleation
        """
        super(BasicBlockBU, self).__init__()
        self.flag_at = opts.model_flag
        self.is_bu2 = is_bu2
        self.ndirections = opts.ndirections
        self.idx = shared.index
        self.ntasks = opts.ntasks
        # computing the shape for the channel and pixel modulation.
        shape_spatial = block_inshapes[1:]
        # computing the shape for the channel and pixel modulation.
        nchannels = block_inshapes[0]
        if task_embedding is None:
            self.task_embedding = [[]]*(opts.ndirections)
        else:
            self.task_embedding = task_embedding
        # TODO - MAKE IT CLEARER.
        # If BU2 stream create the task embedding.
        if self.flag_at is Flag.ZF and self.is_bu2:
            # channel modulation after conv1.
            self.channel_modulation_after_conv1 = Modulation(
                opts=opts, shape=nchannels, pixel_modulation=False, task_embedding=self.task_embedding)
            # pixel modulation after conv1.
            self.pixel_modulation_after_conv1 = Modulation(
                opts=opts, shape=shape_spatial, pixel_modulation=True, task_embedding=self.task_embedding)
            # channel modulation after conv2.
            self.channel_modulation_after_conv2 = Modulation(
                opts=opts, shape=nchannels, pixel_modulation=False, task_embedding=self.task_embedding)
            # pixel modulation after conv2.
            self.pixel_modulation_after_conv2 = Modulation(
                opts=opts, shape=shape_spatial, pixel_modulation=True, task_embedding=self.task_embedding)
        self.conv1 = nn.Sequential(shared.conv1, opts.norm_layer(
            opts, nchannels), opts.activation_fun())  # conv1
        self.conv2 = nn.Sequential(shared.conv2, opts.norm_layer(
            opts, nchannels), opts.activation_fun())  # conv2
        if shared.downsample is not None:
            downsample = nn.Sequential(shared.downsample, opts.norm_layer(opts,
                                                                          nchannels))  # downsample,sometimes needed for the skip connection from the previous block.
        else:
            downsample = None
        self.downsample = downsample
        self.stride = shared.stride
        self.use_lateral = shared.use_lateral
        if self.use_lateral:
            self.lat1 = SideAndComb(opts,
                                    shared.lat1.filters)  # Lateral connection 1 from the previous stream if exists.
            self.lat2 = SideAndComb(opts,
                                    shared.lat2.filters)  # Lateral connection 1 from the previous stream if exists.
            self.lat3 = SideAndComb(opts,
                                    shared.lat3.filters)  # Lateral connection 1 from the previous stream if exists.

    def forward(self, inputs):
        """
        Args:
            inputs: The input, the flag and the lateral connections if exist.

        Returns: The block output, the lateral connections.

        """
        x, flag, laterals_in = inputs  # The inputs are the x(from the previous block or an image) , flag , the lateral connection from the last stream (if exists).
        if laterals_in is not None:
            # There are 3 connections from the last stream.
            lateral1_in, lateral2_in, lateral3_in = laterals_in
        else:
            lateral1_in, lateral2_in, lateral3_in = None, None, None
        laterals_out = []
        if laterals_in is not None:
            x = self.lat1((x, lateral1_in))  # perform lateral skip connection.
        laterals_out.append(x)
        inp = x  # save the inp for the skip to the end of the block
        x = self.conv1(x)  # perform conv
        # perform the task embedding if needed.
        if self.flag_at is Flag.ZF and self.is_bu2:
            flag_ = flag[:, :self.ndirections]
            x = self.pixel_modulation_after_conv1(x, flag_)
            x = self.channel_modulation_after_conv1(x, flag_)

        if laterals_in is not None:  # perform lateral skip connection.
            x = self.lat2((x, lateral2_in))

        laterals_out.append(x)
        x = self.conv2(x)  # perform conv
        # perform the task embedding if needed.
        if self.flag_at is Flag.ZF and self.is_bu2:
            flag_ = flag[:, :self.ndirections]
            x = self.pixel_modulation_after_conv2(x, flag_)
            x = self.channel_modulation_after_conv2(x, flag_)

        if laterals_in is not None:  # perform lateral skip connection.
            x = self.lat3((x, lateral3_in))

        laterals_out.append(x)
        # downsample the input from the beginning of the block to match the expected shape.
        if self.downsample is not None:
            identity = self.downsample(inp)
        else:
            identity = inp

        x = x + identity  # Perform the skip connection.
        return x, laterals_out


class InitialTaskEmbedding(nn.Module):
    """
    The Initial Task embedding at the top of the TD stream.
    The task_embedding is represented by ????
    """

    def __init__(self, opts: argparse, task_embedding) -> None:
        """
        Args:
            opts: The model options.
            task_embedding: The task embedding.
        """
        super(InitialTaskEmbedding, self).__init__()
        self.ntasks = opts.ntasks
        self.top_filters = opts.nfilters[-1]
        self.model_flag = opts.model_flag
        self.ndirections = opts.ndirections
        self.norm_layer = opts.norm_layer
        self.activation_fun = opts.activation_fun
        self.nclasses = opts.nclasses
        self.ds_type = opts.ds_type
        self.train_arg_emb = opts.train_arg_emb
        if self.model_flag is Flag.ZF:
            self.h_flag_task_td = []  # The task embedding.
            self.h_flag_arg_td = []
            self.h_top_td = nn.Sequential(nn.Linear(self.top_filters * 2, self.top_filters),
                                          self.norm_layer(opts, self.top_filters, dims=1), self.activation_fun())
            for i in range(self.ndirections):
                layer = nn.Sequential(nn.Linear(1, self.top_filters // 2),
                                      self.norm_layer(opts, self.top_filters // 2, dims=1), self.activation_fun())
                self.h_flag_task_td.append(layer)
                task_embedding[i].extend(layer.parameters())
            self.h_flag_task_td = nn.ModuleList(self.h_flag_task_td)
            if self.train_arg_emb:
                self.argument_embedding = [[] for _ in range(self.ntasks)]
                for i in range(self.ntasks):  # The argument embedding.
                    layer = nn.Sequential(nn.Linear(self.nclasses[i], self.top_filters // 2),
                                          self.norm_layer(opts, self.top_filters // 2, dims=1), self.activation_fun())
                    self.h_flag_arg_td.append(layer)
                    self.argument_embedding[i].extend(layer.parameters())
                self.h_flag_arg_td = nn.ModuleList(self.h_flag_arg_td)
            else:
                self.h_flag_arg_td = nn.Sequential(nn.Linear(self.nclasses[0], self.top_filters // 2),
                                                   self.norm_layer(
                                                       opts, self.top_filters // 2, dims=1),
                                                   self.activation_fun())

        if self.model_flag is Flag.TD:
            # The task embedding.
            self.h_flag_task_td = nn.Sequential(nn.Linear(self.ndirections, self.top_filters // 2),
                                                self.norm_layer(
                                                    opts, self.top_filters // 2, dims=1),
                                                self.activation_fun())
            # The argument embedding.
            self.h_flag_arg_td = nn.Sequential(nn.Linear(self.nclasses[0], self.top_filters // 2),
                                               self.norm_layer(
                                                   opts, self.top_filters // 2, dims=1),
                                               self.activation_fun())
            # The projection layer.
            self.h_top_td = nn.Sequential(nn.Linear(self.top_filters * 2, self.top_filters),
                                          self.norm_layer(opts, self.top_filters, dims=1), self.activation_fun())

    def forward(self, inputs: tuple) -> tuple:
        """
        Args:
            inputs: The output from BU1 and the flag.

        Returns: The model output.

        """
        (bu_out, flag) = inputs
        direction_flag = flag[:, :self.ndirections]  # The direction vector.
        # The task vector.
        task_flag = flag[:, self.ndirections:self.ndirections + self.ntasks]
        arg = flag[:, self.ndirections + self.ntasks:]  # The argument vector.
        direction_id = flag_to_task(direction_flag)  # The direction id.
        lan_id = flag_to_task(task_flag)  # The lan id.
        ones_ = direction_flag[:, direction_id].view([-1, 1])

        if self.model_flag is Flag.ZF:
            # Take the specific task embedding to avoid forgetting.
            top_td_task = self.h_flag_task_td[direction_id](ones_)
        else:
            top_td_task = self.h_flag_task_td(direction_flag)
        top_td_task = top_td_task.view((-1, self.top_filters // 2, 1, 1))
        if self.train_arg_emb:
            top_td_arg = self.h_flag_arg_td[lan_id](arg)  # Embed the argument.
        else:
            top_td_arg = self.h_flag_arg_td(arg)  # Embed the argument.
        top_td_arg = top_td_arg.view((-1, self.top_filters // 2, 1, 1))
        top_td = torch.cat((top_td_task, top_td_arg),
                           dim=1)  # Concatenate the flags
        top_td_embed = top_td
        h_side_top_td = bu_out
        top_td = torch.cat((h_side_top_td, top_td), dim=1)
        top_td = torch.flatten(top_td, 1)
        top_td = self.h_top_td(top_td)  # The projection layer.
        top_td = top_td.view((-1, self.top_filters, 1, 1))
        x = top_td
        return x, top_td_embed, top_td


class BasicBlockTD(nn.Module):
    # Basic block of the TD stream.
    # The same architecture as in BU just instead of downsampling by stride factor we upsample by stride factor.
    expansion = 1

    def __init__(self, opts, in_channels, out_channels, stride, shape):
        """
        Args:
            opts: The model options.
            in_channels: In channels from the last block.
            out_channels: Out channels for the last block.
            stride: The stride to upsample according.
            shape: The model inshapes.
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
        if stride != 1:
            upsample = nn.Sequential(nn.Upsample(size=size, mode='bilinear', align_corners=False),
                                     conv1x1(in_channels, out_channels, stride=1), opts.norm_layer(opts, out_channels))
        elif in_channels != out_channels:
            upsample = nn.Sequential(conv1x1(
                in_channels, out_channels, stride=1), opts.norm_layer(opts, out_channels))
        self.upsample = upsample

    def forward(self, inputs):
        """
        Args:
            inputs: The Block input, the lateral connections and the flag.

        Returns: The block output, the lateral connections.

        """
        x, flag, laterals_in = inputs
        if laterals_in is not None:
            # There are 3 lateral connections from the last stream
            lateral1_in, lateral2_in, lateral3_in = laterals_in
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
            # Upsample the input if needed,for the skip connection.
            identity = self.upsample(inp)
        else:
            identity = inp
        x = x + identity  # Performs the skip connection
        x = self.relu3(x)
        return x, laterals_out[::-1]
