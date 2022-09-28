import torch
import torch.nn as nn
import argparse
from supp.heads import MultiTaskHead, OccurrenceHead
from supp.blocks import BUInitialBlock, init_module_weights, InitialTaskEmbedding, SideAndComb, SideAndCombSharedBase
from supp.general_functions import depthwise_separable_conv, get_laterals, num_params
from supp.FlagAt import Flag, DsType
import numpy as np


class TDModel(nn.Module):
    def __init__(self, opts: argparse,bu_inshapes:list):
        """
        Args:
            opts: The model opts.
            bu_inshapes: The BU shapes.
        """
        super(TDModel, self).__init__()
        self.block = opts.td_block_type
        self.use_lateral = opts.use_lateral_butd
        self.use_td_flag = opts.use_td_flag
        self.model_flag = opts.model_flag
        self.top_filters = opts.nfilters[-1]
        self.inplanes = opts.nfilters[-1]
        self.opts= opts
        self.ntasks = opts.ntasks
        self.use_td_flag = opts.use_td_flag
        self.inshapes = bu_inshapes
        self.ndirections = opts.ndirections
        upsample_size = opts.avg_pool_size  # before avg pool we have 7x7x512
        self.task_embedding = [[] for _ in range(self.ndirections)]
        self.argument_embedding = [[] for _ in range(self.ntasks)]
        self.InitialTaskEmbedding = InitialTaskEmbedding(opts)
        for i in range(self.ndirections):
            self.task_embedding[i].extend(self.InitialTaskEmbedding.task_embedding[i])
        if opts.ds_type is DsType.Omniglot and self.model_flag is Flag.ZF:
         for j in range(self.ntasks):
          self.argument_embedding[j].extend(self.InitialTaskEmbedding.argument_embedding[j])

        self.top_upsample = nn.Upsample(scale_factor=upsample_size, mode='bilinear',
                                        align_corners=False)  # Upsample layer to make at of the shape before the avgpool.
        layers = []
        for k in range(len(opts.strides) - 1, 0, -1):
            nblocks = opts.ns[k]
            stride = opts.strides[k]
            filters = opts.nfilters[k - 1]
            layers.append(self._make_layer( filters, nblocks, stride=stride,index = k))  # Create the exact opposite layers of the BU1 stream.
        self.alllayers = nn.ModuleList(layers)
        filters = opts.nfilters[0]
        if self.use_lateral:
            self.bot_lat = SideAndComb(opts,filters)
        init_module_weights(self.modules())

    def _make_layer(self,  planes: int, num_blocks: int, stride: int = 1, index:int=0):
        """
        Args:
            planes: The outplanes.
            num_blocks: The number of blocks.
            stride: The stride.
            index: The block index.

        Returns: A ResNet layer.

        """
        layers = []
        for _ in range(1, num_blocks):  # Create shape preserving blocks.
            newblock = self.block(self.opts,self.inplanes, planes, 1,  self.inshapes, index = index)
            layers.append(newblock)
        # Create Upsampling block.
        newblock = self.block(self.opts,self.inplanes, planes, stride,  self.inshapes, index = index)
        layers.append(newblock)
        self.inplanes = planes * self.block.expansion
        return nn.ModuleList(layers)

    def forward(self, inputs: list[torch]) -> list[torch]:
        """
        Args:
            inputs: The output from the BU1 stream, flag the task+arg flag , laterals_in, the laterals from the BU1 stream.

        Returns: The td outputs + lateral connections foR bu1.

        """

        bu_out, flag, laterals_in = inputs
        laterals_out = []
        if self.use_td_flag:
            (x, top_td_embed, top_td) = self.InitialTaskEmbedding((bu_out, flag))  # Compute the initial task embedding.
        else:
            x = bu_out
        laterals_out.append(x)
        x = self.top_upsample(x)  # Upsample to the shape before the avgpooling in the BU1 stream.
        if laterals_in is None or not self.use_lateral:  # If there are not lateral connections or not used.
            for layer in self.alllayers:
                layer_lats_out = []
                for block in layer:
                    x, block_lats_out = block((x, None))
                    layer_lats_out.append(block_lats_out)
                reverse_layer_lats_out = layer_lats_out[::-1]
                laterals_out.append(reverse_layer_lats_out)
        else:  # Here we want to use the lateral connections.
            reverse_laterals_in = laterals_in[::-1]
            for layer, lateral_in in zip(self.alllayers,
                                         reverse_laterals_in[1:-1]):  # Iterating over all layers in the stream.
                layer_lats_out = []  # The lateral connections for the BU2 stream.
                reverse_lateral_in = lateral_in[::-1]  # Inverting the laterals to match the desired shape.
                for block, cur_lat_in in zip(layer, reverse_lateral_in):  # Iterating over all blocks in the layer.
                    reverse_cur_lat_in = cur_lat_in[::-1]  # Inverting the laterals to match the desired shape.
                    x, block_lats_out = block((x, flag,
                                               reverse_cur_lat_in))  # Compute the block output using x, the flag and the lateral connections.
                    layer_lats_out.append(block_lats_out)  # Add the lateral output for the next stream.
                reverse_layer_lats_out = layer_lats_out[::-1]
                laterals_out.append(reverse_layer_lats_out)  # Add all the layer's laterals.
            lateral_in = reverse_laterals_in[-1]
            x = self.bot_lat((x, lateral_in))  # Compute lateral connection + channel modulation.
        laterals_out.append(x)
        outs = [x, laterals_out[::-1]]  # Output the output of the stream + the lateral connections.
        if self.use_td_flag:
            outs += [top_td_embed, top_td]  # Add the top embeddings to the output if needed.
        return outs

class BUStream(nn.Module):
    def __init__(self, opts: argparse, shared: nn.Module, is_bu2: bool):
        """
        Args:
            opts: The model options.
            shared: The shared part between BU1, BU2.
            is_bu2: Whether the stream is BU1 or BU2.
        """
        super(BUStream, self).__init__()
        self.block = opts.bu_block_type
        self.inshapes = shared.inshapes
        self.ntasks = opts.ntasks
        self.ndirections = opts.ndirections
        self.task_embedding = [[] for _ in range(self.ndirections)]
       # self.norm_layer = opts.norm_layer
        self.activation_fun = opts.activation_fun
        self.model_flag = opts.model_flag
        self.inshapes = shared.inshapes_one_list
        self.orig_relus = opts.orig_relus
        self.opts= opts
        self.use_lateral = shared.use_lateral
        self.filters = opts.nfilters[0]
        self.InitialBlock = BUInitialBlock(opts, shared)
        layers = []
        for shared_layer in shared.alllayers:  # For each shared layer we create associate BU layer.
            layers.append(self._make_layer(shared_layer, self.inshapes, is_bu2))
        self.alllayers = nn.ModuleList(layers)
        self.avgpool = shared.avgpool  # Avg pool layer.
        if self.use_lateral:
            self.top_lat = SideAndComb(opts, shared.top_lat.filters)

        init_module_weights(self.modules())

    def _make_layer(self, blocks: nn.Module, inshapes: list, is_bu2: bool) -> nn.ModuleList:
        """
        Args:
            blocks: Shared layers between BU1, BU2.
            inshapes: The input shape of the model.
            is_bu2: Whether BU1 or BU2.

        Returns: A ResNet shared layer.

        """
        layers = []
        for shared_block in blocks:
            # Create Basic BU block.
            layer = self.block(self.opts, shared_block, inshapes, is_bu2,)
            if self.model_flag is Flag.ZF and is_bu2:
                # Adding the task embedding of the BU2 stream.
                for i in range(self.ndirections):
                    self.task_embedding[i].extend(layer.task_embedding[i])
            layers.append(layer)
        return nn.ModuleList(layers)

    def forward(self, inputs: list[torch]) -> tuple:
        """
        Args:
            inputs: The input, the flags, the lateral connection from TD network.

        Returns: The model output + The lateral connections.

        """
        x, flags, laterals_in = inputs  # The input is the image, the flag and the lateral from the previous stream.
        laterals_out = []  # The laterals for the second stream.
        x = self.InitialBlock((x, flags, laterals_in))  # Compute the initial block in ResNet.
        laterals_out.append(x)
        for layer_id, layer in enumerate(self.alllayers):
            layer_lats_out = []
            for block_id, block in enumerate(layer):
                lateral_layer_id = layer_id + 1
                cur_lat_in = get_laterals(laterals_in, lateral_layer_id, block_id)  # Get the laterals associate with the layer,block_id.
                x, block_lats_out = block((x, flags, cur_lat_in))  # Compute the block with the lateral connection.
                layer_lats_out.append(block_lats_out)
            laterals_out.append(layer_lats_out)
        x = self.avgpool(x)  # Avg pool.
        lateral_in = get_laterals(laterals_in, lateral_layer_id + 1, None)
        if self.use_lateral and lateral_in is not None:
            x = self.top_lat((x, lateral_in))  # last lateral connection before the the loss.
        laterals_out.append(x)
        return x, laterals_out

class BUStreamShared(nn.Module):
    def __init__(self, opts: argparse):
        """
        Args:
            opts: Model options.
        """
        super(BUStreamShared, self).__init__()
        block_id = 0
        layers = []
        self.use_lateral = opts.use_lateral_tdbu
        self.block = opts.bu_shared_block_type
        stride = opts.strides[0]
        filters = opts.nfilters[0]
        inplanes = opts.inshape[0]
        self.opts= opts
        inshape = np.array(opts.inshape)
        inshapes = []  # The shapes of all tensors in all blocks.
        inshapes_one_list = []
        self.conv1 = depthwise_separable_conv(inplanes, filters, kernel_size=7, stride=stride, padding=3,   bias=False)  # The first conv layer as in ResNet.
        self.inplanes = filters
        inshape = np.array([filters,np.int(np.ceil(inshape[1] / stride)),np.int(np.ceil(inshape[2] / stride))])  # The first shape.
        inshapes.append([inshape])
        self.bot_lat = SideAndCombSharedBase(filters=filters)
        num_blocks = 0
        for k in range(1, len(opts.strides)):
            nblocks = opts.ns[k]
            stride = opts.strides[k]
            filters = opts.nfilters[k]
            layers.append(self._make_layer(filters, nblocks, stride=stride, num_blocks=num_blocks))
            inshape = np.array([filters,np.int(np.ceil( inshape[1] / stride)),np.int( np.ceil(inshape[2] / stride))])
            inshape_lst = []
            num_blocks = num_blocks + nblocks
            block_id = block_id + nblocks
            for _ in range(nblocks):
                inshape_lst.append(inshape)
                inshapes_one_list.append(inshape)
            inshapes.append(inshape_lst)
        self.alllayers = layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Average pool before the classification.
        filters = opts.nfilters[-1]
        if self.use_lateral:
            self.top_lat = SideAndCombSharedBase(filters=filters)
        inshape = np.array([filters, 1, 1])  # Add the shape of the last layer.
        inshapes.append(inshape)
        self.inshapes = inshapes
        self.inshapes_one_list = inshapes_one_list

    def _make_layer(self, planes: int, nblocks: int, stride: int = 1,num_blocks:int=0) -> nn.Module:
        """
        Args:
            planes: Channels_out of the layer.
            nblocks: num blocks in the layer.
            stride: The stride.
            num_blocks: Num blocks created so far.

        Returns: A ResNet layer.

        """
        layers = []
        layers.append(self.block(self.opts, self.inplanes, planes, stride, num_blocks))  # Add an initial block
        self.inplanes = planes * self.block.expansion
        for idx in range(0, nblocks - 1):  # Add nblocks - 1 preserving blocks.
            layers.append(self.block(self.inplanes, planes, 1, self.use_lateral,num_blocks))
        return layers

class BUModel(nn.Module):
    def __init__(self, opts: argparse,use_embedding:bool):
        """
        Args:
            opts: Model options.
            use_embedding: Whether to use embedding.
        """
        super(BUModel, self).__init__()
        bu_shared = BUStreamShared(opts)
        self.trunk = BUStream(opts, bu_shared, is_bu2=use_embedding)  # In the BUModel there is only BU stream.

    def forward(self, inputs: list[torch]) -> tuple:
        """
        Args:
            inputs: The model inputs.

        Returns: The output + lateral connections

        """
        trunk_out, laterals_out = self.trunk(inputs)
        return trunk_out, laterals_out

# TODO - MAKE BUTDMODEL, BUTDMODELSHARED THE SAME.
class BUTDModel(nn.Module):
    # BU - TD model.
    def forward(self, inputs: list[torch]) -> list[torch]:
        """
        Args:
            inputs: The model input.

        Returns: The output from all streams.

        """
        samples = self.inputs_to_struct(inputs)  # Transform the input to struct.
        images = samples.image
        flags = samples.flag
        model_inputs = [images, flags, None]  # The input to the BU1 stream is just the images and the flags.
        bu_out, bu_laterals_out = self.bumodel1(model_inputs)
        if self.use_bu1_loss:
            occurrence_out = self.occhead(bu_out)  # Compute the occurrence head output.
        else:
            occurrence_out = None
        model_inputs = [bu_out, flags]
        if self.use_lateral_butd:
            model_inputs += [bu_laterals_out]
        else:
            model_inputs += [None]
        td_outs = self.tdmodel(
            model_inputs)  # The input to the TD stream is the bu_out, flags, the lateral connections.
        td_out, td_laterals_out, *td_rest = td_outs
        model_inputs = [images, flags]
        if self.use_lateral_tdbu:
            model_inputs += [td_laterals_out]
        else:
            model_inputs += [[td_out]]
        bu2_out, bu2_laterals_out = self.bumodel2(
            model_inputs)  # The input to the TD stream is the images, flags, the lateral connections.
        head_input = (bu2_out, flags)
        task_out = self.Head(head_input)  # Compute the classification layer.
        outs = [occurrence_out, task_out, bu_out, bu2_out]
        return outs  # Return all the outputs from all streams.

    class outs_to_struct:
        def __init__(self, outs: list[torch]) :
            """
            Struct transforming the model output list to struct.
            Args:
                outs: The model outs.
            """
            occurrence_out, task_out, bu_out, bu2_out = outs
            self.occurence_out = occurrence_out
            self.task = task_out
            self.bu = bu_out
            self.bu2 = bu2_out

    class inputs_to_struct:
        def __init__(self,inputs):
            img, label_task, flag, label_all, label_existence = inputs
            self.image = img
            self.label_all = label_all
            self.label_existence = label_existence
            self.label_task = label_task
            self.flag = flag

class BUTDModelShared(BUTDModel):
    def __init__(self, opts: argparse):
        """
        Args:
            opts: Model options.
        """
        super(BUTDModelShared, self).__init__()
        self.ntasks = opts.ntasks
        self.ndirections = opts.ndirections
        self.use_lateral_butd = opts.use_lateral_butd
        self.use_lateral_tdbu = opts.use_lateral_tdbu
        self.task_embedding = [[] for _ in range(self.ndirections)]  # Container to store the task embedding.
        self.transfer_learning = [[] for _ in range(self.ndirections*self.ntasks)]
        self.argument_embedding = [[] for _ in range(self.ntasks)]
        self.model_flag = opts.model_flag  # The model type
        self.use_bu1_loss = opts.use_bu1_loss  # Whether to use the Occurrence loss.
        self.use_td_flag = opts.use_td_flag
        self.ds_type = opts.ds_type
        if self.use_bu1_loss:
            self.occhead = OccurrenceHead(opts)
        bu_shared = BUStreamShared(opts)  # The shared part between BU1, BU2.
        pre_top_shape = bu_shared.inshapes[-2][-1]
        opts.avg_pool_size = tuple(pre_top_shape[1:].tolist())
        self.bu_inshapes = bu_shared.inshapes
        self.bumodel1 = BUStream(opts, bu_shared, is_bu2=False)  # The BU1 stream.
        self.tdmodel = TDModel(opts,bu_shared.inshapes)  # The TD stream.
        self.bumodel2 = BUStream(opts, bu_shared, is_bu2=True)  # The BU2 stream.
        self.Head = MultiTaskHead(opts)  # The task-head to transform the last layer output to the number of classes.
        if self.model_flag is Flag.ZF:  # Storing the Task embedding.
            for i in range(self.ndirections):
                self.task_embedding[i].extend(self.bumodel2.task_embedding[i])
                self.task_embedding[i].extend(self.tdmodel.task_embedding[i])
            if self.ds_type is DsType.Omniglot:
             for j in range(self.ntasks):
              self.argument_embedding[j].extend(self.tdmodel.argument_embedding[j])
            for i in range(self.ntasks):
             for j in range(self.ndirections):
                  self.transfer_learning[i*self.ndirections+j] =list(self.Head.taskhead[i*self.ndirections+j].parameters() )
        else:
            for i in range(self.ndirections):
                self.task_embedding[i].extend(list(self.Head.taskhead[0][i].parameters()))
class BUModelSimple(nn.Module):
    """
    Only BU network.
    """
    def __init__(self, opts):
        """
        Args:
            opts:  The model options.
        """
        super(BUModelSimple, self).__init__()
        self.taskhead = MultiTaskHead(opts)
        self.bumodel = BUModel(opts,use_embedding=False)
        pre_top_shape = self.bumodel.trunk.inshapes[-2][1:]
        opts.avg_pool_size = tuple(pre_top_shape[1:].tolist())

    def forward(self, inputs):
        samples = self.inputs_to_struct(inputs)
        images = samples.image
        flags = samples.flag
        model_inputs = [images, flags, None]
        bu_out, _ = self.bumodel(model_inputs)
        task_out = self.taskhead((bu_out,flags))
        return task_out, bu_out

    class outs_to_struct:
        def __init__(self, outs: list[torch]) :
            """
            Struct transforming the model output list to struct.
            Args:
                outs: The model outs.
            """
            task_out, bu_out = outs
            self.task = task_out
            self.bu = bu_out

    class inputs_to_struct:
        def __init__(self,inputs):
            img, label_task, flag = inputs
            self.image = img
            self.label_task = label_task
            self.flag = flag
