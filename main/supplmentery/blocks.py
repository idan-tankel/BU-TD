import torch.nn as nn
import torch
import torch._C
import numpy as np
from supplmentery.general_functions import flag_to_task, conv3x3, conv1x1, get_laterals, conv3x3up
from supplmentery.FlagAt import FlagAt
import argparse


class ChannelModulation(nn.Module):
    # The layer performs the channel modulation on the lateral connection
    def __init__(self, nchannels: int) -> None:
        """
        :param nchannels: The number of channels to perform channel-modulation on.
        """
        super(ChannelModulation, self).__init__()
        self.nchannels = nchannels
        shape = [self.nchannels, 1, 1]
        self.weights = nn.Parameter(
            torch.Tensor(*shape))  # creates the learnable parameter of shape [nchannels,1,1] according to nchannels.

    def forward(self, inputs: torch) -> torch:
        """
        :param inputs: receives tensor of shape [N,C,H,W]
        :return: tensor of shape [N,C,H,W]
        """
        return inputs * self.weights  # performs the channel-modulation.


def init_module_weights(modules: nn.Module) -> None:
    # same as our paper's experiments
    """
    Initializes all model layers according to the distributions above.
    :param modules(`nn.Module`): all model's layers
    :return: None

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
    def __init__(self, filters: int, norm_layer: nn.Module, activation_fun: nn.Module, orig_relus: bool,
                 ntasks: int) -> None:
        """
        :param filters: The number of filters in the channel-modulation layer
        :param norm_layer: Batch norm instance
        :param activation_fun: usually relu
        :param orig_relus: whether to use the activation_fun after the batch norm on the lateral connection
        :param ntasks: number of possible tasks
        """
        super(SideAndComb, self).__init__()
        self.side = ChannelModulation(filters)  # channel-modulation layer.
        # batch norm after the channel-modulation of the lateral.
        self.norm = norm_layer(filters, ntasks)
        self.orig_relus = orig_relus
        self.filters = filters
        if not orig_relus:
            self.relu1 = activation_fun()  # activation_fun after the batch_norm layer
        self.relu2 = activation_fun()  # activation_fun after the skip connection

    def forward(self, inputs: torch) -> torch:
        """
        :param inputs: two tensors both of shape [B,C,H,W]
        :return: tensor of shape [B,C,H,W]
        """
        x, lateral = inputs
        side_val = self.side(lateral)  # channel-modulation(CM)
        side_val = self.norm(side_val)  # batch_norm after the CM
        if not self.orig_relus:
            # activation_fun after the batch_norm
            side_val = self.relu1(side_val)
        x = x + side_val  # the lateral skip connection
        x = self.relu2(x)  # activation_fun after the skip connection
        return x


class SideAndCombSharedBase(nn.Module):
    # TODO-understand the difference between them.
    def __init__(self, filters):
        super(SideAndCombSharedBase, self).__init__()
        self.side = ChannelModulation(filters)
        self.filters = filters

# TODO - This can be deleted as there is no shared part between the lateral connections of BU1,BU2


class SideAndCombShared(nn.Module):
    # performs the lateral connection BU1 -> TD or TD -> BU2.
    # Very similar to SideAndComb so the functionality is the same.
    # In this class the channel_modulation is shared between ?
    # TODO-understand where this is shared.

    def __init__(self, shared, norm_layer, activation_func, orig_relus, ntasks):
        super(SideAndCombShared, self).__init__()
        self.side = shared.side
        self.ntasks = ntasks
        self.norm = norm_layer(shared.filters, self.ntasks)
        self.orig_relus = orig_relus
        if not orig_relus:
            self.relu1 = activation_func()
        self.relu2 = activation_func()

    def forward(self, inputs):
        x, lateral = inputs
        side_val = self.side(lateral)
        side_val = self.norm(side_val)
        if not self.orig_relus:
            side_val = self.relu1(side_val)
        x = x + side_val
        x = self.relu2(x)
        return x


class Modulation(nn.Module):  # Modulation layer.
    def __init__(self, inshapes: list, pixel_modulation: bool, ntasks: int) -> None:
        super(Modulation, self).__init__()  # TODO-ask someone about this super
        """
        :param inshapes: shape according to allocate params.
        :param pixel_modulation: whether to perform pixel modulation or channel modulation.
        """
        self.inshapes = inshapes
        self.pixel_modulation = pixel_modulation
        self.task_embedding = [[] for _ in range(ntasks)]
        self.modulation = []
        if self.pixel_modulation:
            # If pixel modulation matches the inner spatial of the input
            self.size = [-1, 1, *inshapes]
        else:
            # If channel modulation matches the number of channels
            self.size = [-1, inshapes, 1, 1]
        inshapes = np.prod(inshapes)
        for i in range(ntasks):  # allocating for every task its task embedding
            layer = nn.Linear(1, inshapes)
            self.task_embedding[i].extend(list(layer.parameters()))
            self.modulation.append(layer)
        self.modulation = nn.ModuleList(self.modulation)

    def forward(self, inputs: torch, flag: torch) -> torch:
        """

        :param inputs: torch of shape [B,C,H,W].
        :param flag: torch of shape [B,S]
        :return: torch of shape [B,C,H,W].
        """
        task_idx = flag_to_task(flag)
        flag_task = flag[:, task_idx].view(-1, 1)
        task_emb = self.modulation[task_idx](flag_task).view(
            self.size)  # computed the task embedding according to the task_idx and changes the shape according to pixel_modulation. # compute the task embedding according to the task_idx.
        inputs = inputs * (1 - task_emb)  # perform the modulation.
        return inputs


class BasicBlockBUShared(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int, use_lateral: bool, idx: int = 0) -> None:
        """
        Basic block of the shared part between BU1,BU2.
        The conv layers of BU1 and BU2 are shared (with the same weights)
        Based at most on the ResNet pytorch's implementation.


        Args:
            in_channels (int): number of input channels
            out_channels (int): number of out channels
            stride (int): the stride of the conv layer
            use_lateral (bool): use lateral connections between BU1 to BU2
            idx (int, optional): _description_. Defaults to 0.
        """
        super(BasicBlockBUShared, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels  # storing variables.
        self.stride = stride
        self.use_lateral = use_lateral
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
    """
    BUInitialBlock represent the first BU block in the BU-TD-BU chain

    Attributes:
        Filters(`int`): number of filters
        norm_layer(): The norm taken
        activation_func(): The activation function
    """

    def __init__(self, opts: argparse, shared: nn.Module) -> None:
        """
        :param opts: The model options. Given by arguments of the user input, parsed by `parser.py` file
        :param shared: This is the shared layers between BU1 BU2. This part contains conv layers
        """
        super(BUInitialBlock, self).__init__()
        self.filters = opts.nfilters[0]
        self.norm_layer = opts.norm_fun
        self.activation_func = opts.activation_fun
        self.orig_relus = opts.orig_relus
        self.ntasks = opts.ntasks
        self.conv1 = nn.Sequential(shared.conv1, opts.norm_fun(self.filters, opts.ntasks),
                                   self.activation_func())  # The initial block downsampling from RGB.
        self.bot_lat = SideAndCombShared(shared.bot_lat, self.norm_layer, self.activation_func, self.orig_relus,
                                         self.ntasks)  # Skip connection from the TD initial embedding.

    def forward(self, inputs: list[torch]) -> torch:
        """
        :param inputs: The images, flags, the lateral connections(if exists).
        :return: The output if the first block.
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
    def __init__(self, shared: nn.Module, norm_layer: nn.Module, activation_fun: nn.Module, inshapes: torch,
                 ntasks: int, flag_at: FlagAt, is_bu2: bool, orig_relus) -> None:
        """
        :param shared: the shared conv layers between BU1, BU2.
        :param norm_layer: an instance of norm layer.
        :param activation_fun: an instance of activation_fun.
        :param inshapes: The input shape of the block.
        :param ntasks: Number of tasks the model should deal with.
        :param flag_at: Flag
        :param is_bu2: is the stream is BU1 or BU2.
        :param orig_relus: whether to use relu after the skip connection.
        """
        super(BasicBlockBU, self).__init__()
        self.orig_relus = orig_relus
        self.flag_at = flag_at
        self.inshapes = inshapes
        self.is_bu2 = is_bu2
        self.idx = shared.index
        self.ntasks = ntasks
        block_inshapes = inshapes[self.idx]
        # computing the shape for the channel and pixel modulation.
        shape_spatial = block_inshapes[1:]
        # computing the shape for the channel and pixel modulation.
        nchannels = block_inshapes[0]
        # If BU2 stream create the task embedding.
        if self.flag_at is FlagAt.SF and self.is_bu2:
            # The parameters stored as task embedding.
            self.task_embedding = [[] for _ in range(ntasks)]
            self.task_embedding_layers = []
            self.channel_modulation_after_conv1 = Modulation(nchannels, False,
                                                             self.ntasks)  # channel modulation after conv1
            self.task_embedding_layers.append(
                self.channel_modulation_after_conv1)
            self.pixel_modulation_after_conv1 = Modulation(shape_spatial, True,
                                                           self.ntasks)  # pixel modulation after conv1
            self.task_embedding_layers.append(
                self.pixel_modulation_after_conv1)
            self.channel_modulation_after_conv2 = Modulation(nchannels, False,
                                                             self.ntasks)  # channel modulation after conv2
            self.task_embedding_layers.append(
                self.channel_modulation_after_conv2)
            self.pixel_modulation_after_conv2 = Modulation(shape_spatial, True,
                                                           self.ntasks)  # pixel modulation after conv2
            self.task_embedding_layers.append(
                self.pixel_modulation_after_conv2)
            for layer in self.task_embedding_layers:  # store for each task its task embedding
                for i in range(ntasks):
                    self.task_embedding[i].extend(layer.task_embedding[i])

        self.conv1 = nn.Sequential(shared.conv1, norm_layer(
            nchannels, self.ntasks), activation_fun())  # conv1
        self.conv2 = nn.Sequential(shared.conv2, norm_layer(
            nchannels, self.ntasks), activation_fun())  # conv2
        if shared.downsample is not None:
            downsample = nn.Sequential(shared.downsample, norm_layer(nchannels,
                                                                     self.ntasks))  # downsample,sometimes needed for the skip connection from the previous block.
        else:
            downsample = None
        self.downsample = downsample
        self.stride = shared.stride
        self.use_lateral = shared.use_lateral
        if self.use_lateral:
            self.lat1 = SideAndCombShared(shared.lat1, norm_layer, activation_fun, self.orig_relus,
                                          self.ntasks)  # Lateral connection 1 from the previous stream if exists.
            self.lat2 = SideAndCombShared(shared.lat2, norm_layer, activation_fun, self.orig_relus,
                                          self.ntasks)  # Lateral connection 1 from the previous stream if exists.
            self.lat3 = SideAndCombShared(shared.lat3, norm_layer, activation_fun, self.orig_relus,
                                          self.ntasks)  # Lateral connection 1 from the previous stream if exists.
        if self.orig_relus:
            self.relu = activation_fun()

    def forward(self, inputs):
        """
        :param inputs: The inputs are: the tensor output(of shape [B,C,H,W] from the previous block or an image, flag of shape[B,S] , the lateral connections from the last stream (if exists),list of 3 tensors.
        :return: tensor
        """
        x, flag, laterals_in = inputs  # The inputs are the x(from the previous block or an image) , flag , the lateral connection from the last stream (if exists).
        if laterals_in is not None:
            # There are 3 connections from the last stream.
            lateral1_in, lateral2_in, lateral3_in = laterals_in
        laterals_out = []
        if laterals_in is not None:
            x = self.lat1((x, lateral1_in))  # perform lateral skip connection.
        laterals_out.append(x)
        inp = x  # save the inp for the skip to the end of the block
        x = self.conv1(x)  # perform conv

        # perform the task embedding if needed.
        if self.flag_at is FlagAt.SF and self.is_bu2:
            x = self.pixel_modulation_after_conv1(x, flag)
            x = self.channel_modulation_after_conv1(x, flag)

        if laterals_in is not None:  # perform lateral skip connection.
            x = self.lat2((x, lateral2_in))

        laterals_out.append(x)
        x = self.conv2(x)  # perform conv

        # perform the task embedding if needed.
        if self.flag_at is FlagAt.SF and self.is_bu2:
            x = self.pixel_modulation_after_conv2(x, flag)
            x = self.channel_modulation_after_conv2(x, flag)

        if laterals_in is not None:  # perform lateral skip connection.
            x = self.lat3((x, lateral3_in))

        laterals_out.append(x)
        # downsample the input from the beginning of the block to match the expected shape.
        if self.downsample is not None:
            identity = self.downsample(inp)
        else:
            identity = inp

        x = x + identity  # Perform the skip connection.
        if self.orig_relus:  # Perform relu activation.
            x = self.relu(x)
        return x, laterals_out


class InitialTaskEmbedding(nn.Module):
    """
    The Initial Task embedding at the top of the TD stream.
    Takes as input the flag and the output from the BU1 stream and returns the task embedded input.
    """

    def __init__(self, opts: argparse) -> None:
        """
        :param opts:Initialize the module according to the opts.
        """
        super(InitialTaskEmbedding, self).__init__()
        self.ntasks = opts.ntasks
        self.top_filters = opts.nfilters[-1]
        self.model_flag = opts.model_flag
        self.use_td_flag = opts.use_td_flag
        self.task_embedding = [[] for _ in range(self.ntasks)]
        self.norm_layer = opts.norm_fun
        self.activation_fun = opts.activation_fun
        self.use_SF = opts.use_SF
        self.nclasses = opts.nclasses
        self.nargs = self.nclasses[0][0]

        if self.model_flag is FlagAt.SF:
            self.h_flag_task_td = []  # The task embedding.
            self.h_top_td = nn.Sequential(nn.Linear(self.top_filters * 2, self.top_filters),  self.norm_layer(
                self.top_filters, dims=1, num_tasks=self.ntasks), self.activation_fun())
            for i in range(self.ntasks):
                layer = nn.Sequential(nn.Linear(1, self.top_filters // 2), self.norm_layer(
                    self.top_filters // 2, dims=1, num_tasks=self.ntasks),  self.activation_fun())
                self.h_flag_task_td.append(layer)
                self.task_embedding[i].extend(layer.parameters())
            self.h_flag_arg_td = nn.Sequential(nn.Linear(self.nargs, self.top_filters // 2), self.norm_layer(
                self.top_filters // 2, dims=1, num_tasks=self.ntasks),  self.activation_fun())
            self.h_flag_task_td = nn.ModuleList(self.h_flag_task_td)
            # The argument embedding.

            # The projection layer.

        if self.model_flag is FlagAt.TD:
            # The task embedding.
            self.h_flag_task_td = nn.Sequential(nn.Linear(self.ntasks, self.top_filters // 2),
                                                self.norm_layer(
                                                    self.top_filters // 2, dims=1, num_tasks=self.ntasks),
                                                self.activation_fun())
            # The argument embedding.
            self.h_flag_arg_td = nn.Sequential(nn.Linear(opts.nargs, self.top_filters // 2),
                                               self.norm_layer(
                                                   self.top_filters // 2, dims=1, num_tasks=self.ntasks),
                                               self.activation_fun())
            # The projection layer.
            self.h_top_td = nn.Sequential(nn.Linear(self.top_filters * 2, self.top_filters),
                                          self.norm_layer(
                                              self.top_filters, dims=1, num_tasks=self.ntasks),
                                          self.activation_fun())

    def forward(self, inputs: tuple) -> tuple:
        """
        :param inputs: bu_out-the output from the BU1 stream, flag the task and argument one hot vectors.
        :return: The initial task embedding to the top of the TD network.
        """
        (bu_out, flag) = inputs
        task = flag[:, :self.ntasks]  # The task vector.
        arg = flag[:, self.ntasks:]  # The argument vector.
        task_id = flag_to_task(flag)
        flag_task = flag[:, task_id].view(-1, 1)
        if self.use_SF:
            # Take the specific task embedding to avoid forgetting.
            top_td_task = self.h_flag_task_td[task_id](flag_task)
        else:
            top_td_task = self.h_flag_task_td(task)
        top_td_task = top_td_task.view((-1, self.top_filters // 2, 1, 1))
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

    def __init__(self, in_channels, out_channels, stride, norm_layer, activation_fun, use_lateral, inshapes, ntasks,
                 orig_relus, index):
        """
        :param in_channels: in channels from the last block.
        :param out_channels: the out channels of the block.
        :param stride: the stride to upsample according
        :param norm_layer: an instance of norm layer
        :param activation_fun: an instance of activation fun
        :param use_lateral: whether to perform lateral connection from the last stream
        :param ntasks: Number of tasks the model will deal with
        :param orig_relus: whether to use relu after the skip connection.
        """
        super(BasicBlockTD, self).__init__()
        self.ntasks = ntasks
        self.orig_relus = orig_relus
        self.flag_params = [[] for _ in range(self.ntasks)]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inshapes = inshapes
        self.stride = stride
        size = tuple(self.inshapes[index - 1][0][1:])
        if use_lateral:
            self.lat1 = SideAndComb(
                in_channels, norm_layer, activation_fun, orig_relus, ntasks)
            self.lat2 = SideAndComb(
                in_channels, norm_layer, activation_fun, orig_relus, ntasks)
            self.lat3 = SideAndComb(
                out_channels, norm_layer, activation_fun, orig_relus, ntasks)
        self.conv1 = conv3x3(in_channels, in_channels)
        self.conv1_norm = norm_layer(in_channels, self.ntasks)
        self.relu1 = activation_fun()
        self.conv2 = conv3x3up(in_channels, out_channels, size, stride)
        self.conv2_norm = norm_layer(out_channels, self.ntasks)
        if self.orig_relus:
            self.relu2 = activation_fun()
        self.relu3 = activation_fun()
        upsample = None
        out_channels = out_channels * BasicBlockTD.expansion
        if stride != 1:
            upsample = nn.Sequential(nn.Upsample(size=size, mode='bilinear', align_corners=False),
                                     conv1x1(in_channels, out_channels, stride=1), norm_layer(out_channels, ntasks))
        elif in_channels != out_channels:
            upsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride=1), norm_layer(out_channels, ntasks))
        self.upsample = upsample

    def forward(self, inputs):
        """
        :param inputs: The inputs are: the tensor output(of shape from the previous block or an image, flag, the lateral connections from the last stream (if exists),list of 3 tensors.
        :return: tensor
        """
        x, flag, laterals_in = inputs
        if laterals_in is not None:
            # There are 3 lateral connections from the last stream
            lateral1_in, lateral2_in, lateral3_in = laterals_in
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
        if self.orig_relus:
            x = self.relu2(x)
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
