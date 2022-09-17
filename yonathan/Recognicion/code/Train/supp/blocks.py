import torch.nn as nn
import torch
import numpy as np
from supp.general_functions import conv3x3, conv1x1, conv3x3up, get_laterals, flag_to_task,num_params
from supp.FlagAt import Flag ,DsType
import argparse

class ChannelModulation(nn.Module):
    # The layer performs the channel modulation on the lateral connection
    def __init__(self, nchannels: int) -> None:
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
        Args:
            inputs: receives tensor of shape [N,C,H,W].

        Returns: tensor of shape [N,C,H,W].
        """
        return inputs * self.weights  # performs the channel-modulation.

def init_module_weights(modules: nn.Module) -> None:
    # same as our paper's experiments
    # Initializes all model layers according to the distributions above.
    """
    Args:
        modules: all model's layers.
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
    def __init__(self,opts, filters: int ):
        """
        Args:
            opts: The model options.
            filters: The number of filters to do the modulation on.
        """
        super(SideAndComb, self).__init__()
        self.side = ChannelModulation(filters)  # channel-modulation layer.
        self.norm = opts.norm_fun(opts,filters)  # batch norm after the channel-modulation of the lateral.
        self.orig_relus = opts. orig_relus
        self.filters = filters
        if not opts.orig_relus:
            self.relu1 = opts.activation_fun()  # activation_fun after the batch_norm layer
        self.relu2 = opts.activation_fun()  # activation_fun after the skip connection

    def forward(self, inputs: torch) -> torch:
        """
        :param inputs: two tensors both of shape [B,C,H,W]
        :return: tensor of shape [B,C,H,W]
        """
        x, lateral = inputs
        side_val = self.side(lateral)  # channel-modulation(CM)
        side_val = self.norm(side_val)  # batch_norm after the CM
        if not self.orig_relus:
            side_val = self.relu1(side_val)  # activation_fun after the batch_norm
        x = x + side_val  # the lateral skip connection
        x = self.relu2(x)  # activation_fun after the skip connection
        return x
    
'''
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

    def __init__(self,opts, shared):
        super(SideAndCombShared, self).__init__()
        self.side = shared.side
        self.ntasks = opts.ntasks
        self.norm = opts.norm_fun(opts,shared.filters)
        self.orig_relus = opts.orig_relus
        if not opts.orig_relus:
            self.relu1 = opts.activation_fun()
        self.relu2 = opts.activation_fun()

    def forward(self, inputs):
        x, lateral = inputs
        side_val = self.side(lateral)
        side_val = self.norm(side_val)
        if not self.orig_relus:
            side_val = self.relu1(side_val)
        x = x + side_val
        x = self.relu2(x)
        return x
'''

class Modulation(nn.Module):  # Modulation layer.
    def __init__(self, inshapes: list, pixel_modulation: bool, ntasks: int) -> None:
        """
        Args:
            inshapes: shape according to allocate params.
            pixel_modulation: whether to perform pixel modulation or channel modulation.
            ntasks: number of tasks.
        """
        super(Modulation, self).__init__()  # TODO-ask someone about this super
        self.inshapes = inshapes
        self.pixel_modulation = pixel_modulation
        self.task_embedding = [[] for _ in range(ntasks)]
        self.modulation = []
        if self.pixel_modulation:
            size = inshapes[1:]
            self.size = [-1, 1, *size]  # If pixel modulation matches the inner spatial of the input
        else:
            size = inshapes[0]
            self.size = [-1, size, 1, 1]  # If channel modulation matches the number of channels
        inshapes = np.prod(size)
       # bias = False
        for i in range(ntasks):  # allocating for every task its task embedding
            layer = nn.Linear(1,inshapes,True)
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
        task_emb = self.modulation[task_idx](flag_task).view( self.size)  # computed the task embedding according to the task_idx and changes the shape according to pixel_modulation. # compute the task embedding according to the task_idx.
        inputs = inputs * (1 - task_emb)  # perform the modulation.
        return inputs

class BasicBlockBUShared(nn.Module):
    # Basic block of the shared part between BU1,BU2.
    # Based at most on the ResNet pytorch's implementation.
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int, use_lateral: bool,idx:int=0) -> None:
        """
        Args:
            in_channels: in channel from the previous block.
            out_channels: out channel of the block for the Next block.
            stride: stride to perform.
            use_lateral: whether to perform the lateral connection from the previous stream.
            idx: The block index.
        """
        super(BasicBlockBUShared, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels  # storing variables.
        self.stride = stride
        self.use_lateral = use_lateral
        self.index = idx
        self.conv1 = conv3x3(in_channels, out_channels, stride)  # changes the number of channels and the spatial shape.
        self.conv2 = conv3x3(out_channels, out_channels)  # preserves the tensor shape
        downsample = None
        if stride != 1 or in_channels != out_channels * BasicBlockBUShared.expansion:
            downsample = conv1x1(in_channels, out_channels * BasicBlockBUShared.expansion, stride)  # performs downsmaple on the lateral connection to match the shape after conv1.
        self.downsample = downsample
        if self.use_lateral:
            self.lat1 = SideAndCombSharedBase(filters=in_channels)  # The lateral connection from the previous stream.
            self.lat2 = SideAndCombSharedBase(filters=out_channels)  # The lateral connection from the previous stream.
            self.lat3 = SideAndCombSharedBase(filters=out_channels)  # The lateral connection from the previous stream.

class BUInitialBlock(nn.Module):
    def __init__(self, opts: argparse, shared: nn.Module) -> None:
        """
        :param opts: The model options.
        :param shared: The shared part between BU1, BU2.
        """
        super(BUInitialBlock, self).__init__()
        self.filters = opts.nfilters[0]
        self.norm_layer = opts.norm_fun
        self.activation_fun = opts.activation_fun
        self.orig_relus = opts.orig_relus
        self.ntasks = opts.ntasks
        self.conv1 = nn.Sequential(shared.conv1, opts.norm_fun(opts,self.filters), self.activation_fun())  # The initial block downsampling from RGB.
        self.bot_lat = SideAndCombShared(opts,shared.bot_lat)  # Skip connection from the TD initial embedding.

    def forward(self, inputs: list[torch]) -> torch:
        """
        :param inputs: The images, flags, the lateral connections(if exists).
        :return: The output if the first block.
        """
        x, flags, laterals_in = inputs  # The input is the image, the flag and the lateral from the previous stream.
        x = self.conv1(x)  # Compute conv1.
        lateral_in = get_laterals(laterals_in, 0, None)  # The initial lateral connection.
        if lateral_in is not None:
            x = self.bot_lat((x, lateral_in))  # Compute the skip connection.
        return x

def create_task_embedding(shape_first_block,shape_second_block,ntasks):
    task_embedding_layers = []
    channel_modulation_after_conv1 = Modulation(shape_first_block, False, ntasks)  # channel modulation after conv1
    task_embedding_layers.append(channel_modulation_after_conv1)
    pixel_modulation_after_conv1 = Modulation(shape_first_block, True, ntasks)  # pixel modulation after conv1
    task_embedding_layers.append(pixel_modulation_after_conv1)
    channel_modulation_after_conv2 = Modulation(shape_second_block, False, ntasks)  # channel modulation after conv2
    task_embedding_layers.append(channel_modulation_after_conv2)
    pixel_modulation_after_conv2 = Modulation(shape_second_block, True, ntasks)  # pixel modulation after conv2
    task_embedding_layers.append(pixel_modulation_after_conv2)
    embedding = nn.ModuleList([channel_modulation_after_conv1,pixel_modulation_after_conv1, channel_modulation_after_conv2, pixel_modulation_after_conv2])
    return embedding


class BasicBlockBU(nn.Module):
    # Basic block of the BU1,BU2 streams.
    def __init__(self,opts, shared: nn.Module,inshapes:list, is_bu2: bool) -> None:
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
        self.orig_relus = opts.orig_relus
        self.flag_at = opts.model_flag
        self.opts = opts
        self.inshapes = inshapes
        self.is_bu2 = is_bu2
        self.idx = shared.index
        self.ntasks = opts.ntasks
        norm_layer = opts.norm_fun
        activation_fun = opts.activation_fun
        nchannels = inshapes[self.idx+1][0]  # computing the shape for the channel and pixel modulation.
        ndirections = opts.ndirections
        self.use_double_emb = opts.use_double_emb
        self.nembs = 2 if self.use_double_emb else 1
        self.lang_embedding = [[] for _ in range(self.ntasks)]
        self.direction_embedding = [[] for _ in range(ndirections)]

        if self.flag_at is Flag.SF and self.is_bu2:  # If BU2 stream create the task embedding.
           shape_first_block = inshapes[self.idx + 1]
           shape_second_block = inshapes[self.idx + 1]
           self.original_embedding = create_task_embedding(shape_first_block,shape_second_block,self.ntasks)
           for i in range(self.ntasks):
             self.lang_embedding[i].extend(self.original_embedding[0].task_embedding[i])
             self.lang_embedding[i].extend(self.original_embedding[1].task_embedding[i])
             self.lang_embedding[i].extend(self.original_embedding[2].task_embedding[i])
             self.lang_embedding[i].extend(self.original_embedding[3].task_embedding[i])

        if self.flag_at is Flag.SF and self.is_bu2 and self.use_double_emb :
            shape_first_block = inshapes[self.idx]
            shape_second_block = inshapes[self.idx + 1]
            self.second_embedding = create_task_embedding(shape_first_block, shape_second_block,  self.ntasks)
            for i in range(ndirections):
              self.direction_embedding[i].extend(self.second_embedding[0].task_embedding[i])
              self.direction_embedding[i].extend(self.second_embedding[1].task_embedding[i])
              self.direction_embedding[i].extend(self.second_embedding[2].task_embedding[i])
              self.direction_embedding[i].extend(self.second_embedding[3].task_embedding[i])

        self.conv1 = nn.Sequential(shared.conv1, norm_layer(opts,nchannels), activation_fun())  # conv1
        self.conv2 = nn.Sequential(shared.conv2, norm_layer(opts,nchannels), activation_fun())  # conv2
        if shared.downsample is not None:
            downsample = nn.Sequential(shared.downsample, norm_layer(opts, nchannels))  # downsample,sometimes needed for the skip connection from the previous block.
        else:
            downsample = None
        self.downsample = downsample
        self.stride = shared.stride
        self.use_lateral = shared.use_lateral
        if self.use_lateral:
            self.lat1 = SideAndCombShared(opts,shared.lat1)  # Lateral connection 1 from the previous stream if exists.
            self.lat2 = SideAndCombShared(opts,shared.lat2)  # Lateral connection 1 from the previous stream if exists.
            self.lat3 = SideAndCombShared(opts,shared.lat3)  # Lateral connection 1 from the previous stream if exists.
        if self.orig_relus:
            self.relu = activation_fun()

    def forward(self, inputs):
        """
        :param inputs: The inputs are: the tensor output(of shape [B,C,H,W] from the previous block or an image, flag of shape[B,S] , the lateral connections from the last stream (if exists),list of 3 tensors.
        :return: tensor
        """
        x, flag, laterals_in = inputs  # The inputs are the x(from the previous block or an image) , flag , the lateral connection from the last stream (if exists).
        if laterals_in is not None:
            lateral1_in, lateral2_in, lateral3_in = laterals_in  # There are 3 connections from the last stream.
        laterals_out = []
        if laterals_in is not None:
            x = self.lat1((x, lateral1_in))  # perform lateral skip connection.
        laterals_out.append(x)
        inp = x  # save the inp for the skip to the end of the block

        if self.flag_at is Flag.SF and self.is_bu2 and self.nembs > 1:
            flag_direction = flag[:, flag.shape[1] - 2:]
            x = self.second_embedding[0](x, flag_direction)
            x = self.second_embedding[1](x, flag_direction)

        x = self.conv1(x)  # perform conv

        if self.flag_at is Flag.SF and self.is_bu2:  # perform the task embedding if needed.
            lan_flag = flag[:, :self.ntasks]  # The task vector.
            x = self.original_embedding[0](x, lan_flag)
            x = self.original_embedding[1](x, lan_flag)

        if laterals_in is not None:  # perform lateral skip connection.
            x = self.lat2((x, lateral2_in))

        if self.flag_at is Flag.SF and self.is_bu2 and self.nembs > 1:
            flag_direction = flag[:, flag.shape[1] - 2:]
            x = self.second_embedding[2](x, flag_direction)
            x = self.second_embedding[3](x, flag_direction)

        laterals_out.append(x)
        x = self.conv2(x)  # perform conv
        if self.flag_at is Flag.SF and self.is_bu2:  # perform the task embedding if needed.
            lan_flag = flag[:, :self.ntasks]  # The task vector.
            x = self.original_embedding[2](x, lan_flag)
            x = self.original_embedding[3](x, lan_flag)

        if laterals_in is not None:  # perform lateral skip connection.
            x = self.lat3((x, lateral3_in))


        laterals_out.append(x)
        if self.downsample is not None:  # downsample the input from the beginning of the block to match the expected shape.
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
        self.ndirections = opts.ndirections
        self.lang_embedding = [[] for _ in range(self.ntasks)]
        self.direction_embedding = [[] for _ in range(self.ntasks)]
        self.norm_layer = opts.norm_fun
        self.activation_fun = opts.activation_fun
        self.nclasses = opts.nclasses
        self.train_arg_emb = opts.ds_type is DsType.Omniglot
        self.nheads = 3 if opts.use_double_emb else 2
        self.use_double_emb = opts.use_double_emb
        if self.model_flag is Flag.SF:
            self.h_flag_task_td = []  # The task embedding.
            self.h_flag_arg_td = []
            self.h_top_td = nn.Sequential(nn.Linear(2 * self.top_filters, self.top_filters),  self.norm_layer(opts,self.top_filters, dims=1), self.activation_fun())

            for i in range(self.ntasks):
                layer = nn.Sequential(nn.Linear(1, self.top_filters //self.nheads ), self.norm_layer(opts,self.top_filters //self.nheads, dims=1),  self.activation_fun())
                self.h_flag_task_td.append(layer)
                self.lang_embedding[i].extend(layer.parameters())

            if self.train_arg_emb:
             self.argument_embedding = [[] for _ in range(self.ntasks)]
             for i in range(self.ntasks):
                layer = nn.Sequential(nn.Linear(self.nclasses[i], self.top_filters //self.nheads ), self.norm_layer(opts,self.top_filters//self.nheads , dims=1),  self.activation_fun())
                self.h_flag_arg_td.append(layer)
                self.argument_embedding[i].extend(layer.parameters())
             self.h_flag_arg_td = nn.ModuleList(self.h_flag_arg_td)

             if self.use_double_emb:
              self.direction_td = []
              for i in range(self.ndirections):
                 layer = nn.Sequential(nn.Linear(1, self.top_filters //self.nheads ),  self.norm_layer(opts,self.top_filters//self.nheads, dims=1),  self.activation_fun())
                 self.direction_td.append(layer)
                 self.direction_embedding[i].extend(list(layer.parameters()))
              self.direction_td = nn.ModuleList(self.direction_td)

            else:
             self.h_flag_arg_td = nn.Sequential(nn.Linear(self.nclasses[0], self.top_filters // 2), self.norm_layer(self.top_filters // 2, dims=1, num_tasks=self.ntasks),  self.activation_fun())
            self.h_flag_task_td = nn.ModuleList(self.h_flag_task_td)

            # The argument embedding.

            # The projection layer.
        if self.model_flag is Flag.TD:
            # The task embedding.
            self.h_flag_task_td = nn.Sequential(nn.Linear(self.ntasks, self.top_filters // 2), self.norm_layer(self.top_filters // 2, dims=1, num_tasks=self.ntasks), self.activation_fun())
            # The argument embedding.
            self.h_flag_arg_td = nn.Sequential(nn.Linear(opts.nargs, self.top_filters // 2), self.norm_layer(self.top_filters // 2, dims=1, num_tasks=self.ntasks), self.activation_fun())
            # The projection layer.
            self.h_top_td = nn.Sequential(nn.Linear(self.top_filters * 2, self.top_filters), self.norm_layer(self.top_filters, dims=1, num_tasks=self.ntasks), self.activation_fun())

    def forward(self, inputs: tuple) -> tuple:
        """
        :param inputs: bu_out-the output from the BU1 stream, flag the task and argument one hot vectors.
        :return: The initial task embedding to the top of the TD network.
        """
        (bu_out, flag) = inputs
        lan_flag = flag[:, :self.ntasks]  # The task vector.
        arg_flag = flag[:, self.ntasks:flag.shape[1] - self.ndirections]  # The argument vector.
        direction_flag = flag[:,flag.shape[1]-2:]

        lan_id = flag_to_task(lan_flag)
        direction_id = flag_to_task(direction_flag)

        lan_ones = flag[:, lan_id].view(-1, 1)

        direction_ones = flag[:, direction_id].view(-1, 1)

        if self.model_flag is Flag.SF:
            top_td_task = self.h_flag_task_td[lan_id](lan_ones)  # Take the specific task embedding to avoid forgetting.
        else:
            top_td_task = self.h_flag_task_td(task)
        if self.use_double_emb :
            top_td_direction = self.direction_td[direction_id](direction_ones).view((-1, self.top_filters //self.nheads, 1, 1))
        top_td_task = top_td_task.view((-1, self.top_filters // self.nheads , 1, 1))
        if self.train_arg_emb:
         top_td_arg = self.h_flag_arg_td[lan_id](arg_flag)  # Embed the argument.
        else:
         top_td_arg = self.h_flag_arg_td(arg)  # Embed the argument.
        top_td_arg = top_td_arg.view((-1, self.top_filters // self.nheads , 1, 1))
        top_td = torch.cat((top_td_task, top_td_arg), dim = 1)  # Concatenate the flags
        if self.use_double_emb:
            top_td = torch.cat((top_td, top_td_direction),dim = 1)
        top_td_embed = top_td
        h_side_top_td = bu_out
        top_td = torch.cat((h_side_top_td, top_td), dim=1)
        top_td = torch.flatten(top_td, 1)
    #    print(top_td.shape)
        top_td = self.h_top_td(top_td)  # The projection layer.
        top_td = top_td.view((-1, self.top_filters, 1, 1))
        x = top_td
        return x, top_td_embed, top_td

class BasicBlockTD(nn.Module):
    # Basic block of the TD stream.
    # The same architecture as in BU just instead of downsampling by stride factor we upsample by stride factor.
    expansion = 1

    def __init__(self,opts,in_channels, out_channels, stride, inshapes,index,use_lateral):
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
        self.ntasks = opts.ntasks
        self.orig_relus = opts.orig_relus
        self.flag_params = [[] for _ in range(self.ntasks)]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inshapes = inshapes
        self.stride = stride
        size = tuple(self.inshapes[index - 1][0][1:])

        if use_lateral:
            self.lat1 = SideAndComb(opts, in_channels)
            self.lat2 = SideAndComb(opts, in_channels)
            self.lat3 = SideAndComb(opts, out_channels)
        self.conv1 = conv3x3(in_channels, in_channels)
        self.conv1_norm = opts.norm_fun(opts,in_channels)
        self.relu1 = opts.activation_fun()
        self.conv2 = conv3x3up(in_channels, out_channels,size, stride)
        self.conv2_norm = opts.norm_fun(opts, out_channels)
        if self.orig_relus:
            self.relu2 = opts.activation_fun()
        self.relu3 = opts.activation_fun()
        upsample = None
        out_channels = out_channels * BasicBlockTD.expansion
        if stride != 1:
            upsample = nn.Sequential(nn.Upsample(size = size, mode='bilinear', align_corners=False),  conv1x1(in_channels, out_channels, stride=1), opts.norm_fun(opts,out_channels))
        elif in_channels != out_channels:
            upsample = nn.Sequential(conv1x1(in_channels, out_channels, stride=1), opts.norm_fun(opts, out_channels))
        self.upsample = upsample

    def forward(self, inputs):
        """
        :param inputs: The inputs are: the tensor output(of shape from the previous block or an image, flag, the lateral connections from the last stream (if exists),list of 3 tensors.
        :return: tensor
        """
        x, flag, laterals_in = inputs
        if laterals_in is not None:
            lateral1_in, lateral2_in, lateral3_in = laterals_in  # There are 3 lateral connections from the last stream
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
            identity = self.upsample(inp)  # Upsample the input if needed,for the skip connection.
        else:
            identity = inp
        x = x + identity  # Performs the skip connection
        x = self.relu3(x)
        return x, laterals_out[::-1]
