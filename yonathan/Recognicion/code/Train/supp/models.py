import torch
import torch.nn as nn
import argparse
from supp.heads import MultiTaskHead, OccurrenceHead
from supp.blocks import BUInitialBlock, init_module_weights, InitialTaskEmbedding, SideAndComb
from supp.general_functions import depthwise_separable_conv, get_laterals, num_params
from supp.FlagAt import Flag, DsType
from types import SimpleNamespace
import numpy as np

class TDModel(nn.Module):
    def __init__(self, opts: argparse,bu_inshapes:list):
        """
        Args:
            opts: The model options.
            bu_inshapes: The inshapes of the BU model for the up-sampling.
        """

        super(TDModel, self).__init__()
        self.block = opts.td_block_type
        self.use_lateral = opts.use_lateral_butd
        self.activation_fun = opts.activation_fun
        self.use_td_flag = opts.use_td_flag
        self.model_flag = opts.model_flag
        self.orig_relus = opts.orig_relus
        self.norm_layer = opts.norm_fun
        self.top_filters = opts.nfilters[-1]
        self.inplanes = opts.nfilters[-1]
        self.ntasks = opts.ntasks
        self.use_td_flag = opts.use_td_flag
        self.opts = opts
        self.ndirections = opts.ndirections
        self.inshapes = bu_inshapes
        self.upsample_size = opts.avg_pool_size  # before avg pool we have 7x7x512
        self.direction_embedding = [[] for _ in range(self.ntasks)]
        self.lang_embedding = [[] for _ in range(self.ntasks)]
        self.argument_embedding = [[] for _ in range(self.ntasks)]
        self.InitialTaskEmbedding = InitialTaskEmbedding(opts)
        for i in range(self.ntasks):
            self.lang_embedding[i].extend(self.InitialTaskEmbedding.lang_embedding[i])
            if opts.ds_type is DsType.Omniglot:
             self.argument_embedding[i].extend(self.InitialTaskEmbedding.argument_embedding[i])
        for i in range(self.ndirections):
            self.direction_embedding[i].extend(self.InitialTaskEmbedding.direction_embedding[i])
        self.top_upsample = nn.Upsample(scale_factor = self.upsample_size, mode='bilinear',  align_corners=False)  # Upsample layer to make at of the shape before the avgpool.
        layers = []
        for k in range(len(opts.strides) - 1, 0, -1):
            nblocks = opts.ns[k]
            stride = opts.strides[k]
            filters = opts.nfilters[k - 1]
            layers.append(self._make_layer(filters, nblocks, stride=stride, index = k))  # Create the exact opposite layers of the BU1 stream.
        self.alllayers = nn.ModuleList(layers)
        filters = opts.nfilters[0]
        if self.use_lateral:
            self.bot_lat = SideAndComb(opts,filters)
        init_module_weights(self.modules())

    def _make_layer(self, planes: int, num_blocks: int, stride: int = 1, index:int=0):
        """
        Args:
            planes: The out planes of the block.
            num_blocks: Number of identical blocks in the layer.
            stride: The stride of the last block in the layer.
            index: The index of the block.

        Returns: A ResNet layer.

        """
        layers = []
        for _ in range(1, num_blocks):  # Create shape preserving blocks.
            newblock = self.block(self.opts, self.inplanes, 1, stride, self.inshapes, index=index, use_lateral=self.use_lateral)
            layers.append(newblock)
        # Create Upsampling block.
        newblock = self.block(self.opts,self.inplanes, planes, stride, self.inshapes, index = index)
        layers.append(newblock)
        self.inplanes = planes * self.block.expansion
        return nn.ModuleList(layers)

    def forward(self, inputs: list[torch]) -> list[torch]:
        """
        Args:
            inputs: The output from the last layer if exists, lateral connections, and the flag.

        Returns: The TD-Model outs.
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
                    x, block_lats_out = block((x, flag, reverse_cur_lat_in))  # Compute the block output using x, the flag and the lateral connections.
                    layer_lats_out.append(block_lats_out)  # Add the lateral output for the next stream.
                reverse_layer_lats_out = layer_lats_out[::-1]
                laterals_out.append(reverse_layer_lats_out)  # Add all the layer's laterals.
            lateral_in = reverse_laterals_in[-1]
            x = self.bot_lat((x, lateral_in))  # Compute lateral connection + channel modulation.
    #    if self.use_final_conv:  # Compute the last conv layer.
     #       x = self.conv1(x)
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
            is_bu2: Whether the stream is BU1, or BU2.
        """

        super(BUStream, self).__init__()
        self.block = opts.bu_block_type
        self.inshapes = shared.inshapes_devided_by_blocks
        self.opts = opts
        self.ndirections = opts.ndirections
        self.ntasks = opts.ntasks
        self.direction_embedding = [[] for _ in range(self.ndirections)]
        self.lang_embedding = [[] for _ in range(self.ntasks)]
        self.model_flag = opts.model_flag
        self.inshapes = shared.all_shapes_in_one_list
        self.orig_relus = opts.orig_relus
        self.use_lateral = shared.use_lateral
        self.filters = opts.nfilters[0]
        self.is_bu2 = is_bu2
        self.InitialBlock = BUInitialBlock(opts, shared)
        layers = []
        for shared_layer in shared.alllayers:  # For each shared layer we create associate BU layer.
            layers.append(self._make_layer(shared_layer, self.inshapes))
        self.alllayers = nn.ModuleList(layers)
        self.avgpool = shared.avgpool  # Avg pool layer.
        if self.use_lateral and is_bu2:
            self.top_lat = SideAndComb(opts,shared.top_lat_shape)
        init_module_weights(self.modules())

    def _make_layer(self, blocks: nn.Module, inshapes: list) -> nn.ModuleList:
        """
        Args:
            blocks: The Shared block between BU1, BU2.
            inshapes: The layer inshape.

        Returns:

        """
        layers = []
        for shared_block in blocks:
            # Create Basic BU block.
            layer = self.block(self.opts, shared_block,inshapes, self.is_bu2)
            if self.model_flag is Flag.SF and self.is_bu2:
                # Adding the task embedding of the BU2 stream.
                for i in range(self.ntasks):
                    self.lang_embedding[i].extend(layer.lang_embedding[i])
                for i in range(self.ndirections):
                  self.direction_embedding[i].extend(layer.direction_embedding[i])
            layers.append(layer)
        return nn.ModuleList(layers)

    def forward(self, inputs: list[torch]) -> list[torch]:
        """
        Args:
            inputs: The Image, the flag and the lateral connections if exist.

        Returns: The stream output and the lateral connections.
        """

        x, flags, laterals_in = inputs  # The input is the image, the flag and the lateral from the previous stream.
        laterals_out = []  # The laterals for the second stream.
        x = self.InitialBlock((x, flags, laterals_in))  # Compute the initial block in ResNet.
        laterals_out.append(x)
        for layer_id, layer in enumerate(self.alllayers):
            layer_lats_out = []
            for block_id, block in enumerate(layer):
                lateral_layer_id = layer_id + 1
                cur_lat_in = get_laterals(laterals_in, lateral_layer_id,  block_id)  # Get the laterals associate with the layer,block_id.
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
            opts: The model options.
        """

        super(BUStreamShared, self).__init__()
        block_id = 0
        layers = []
        self.activation_fun = opts.activation_fun
        self.use_lateral = opts.use_lateral_tdbu
        self.block = opts.bu_shared_block_type
        stride = opts.strides[0]
        filters = opts.nfilters[0]
        inplanes = opts.inshape[0]
        inshape = np.array(opts.inshape)
        inshapes = []  # The shapes of all tensors in all blocks.
        self.conv1 = depthwise_separable_conv(inplanes, filters, kernel_size=7, stride=stride, padding=3,   bias=False)  # The first conv layer as in ResNet.
        self.inplanes = filters
        inshape = np.array([filters,np.int(np.ceil(inshape[1] / stride)),np.int(np.ceil(inshape[2] / stride))])  # The first shape.
        inshapes.append([inshape])
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
            inshapes.append(inshape_lst)
        self.alllayers = layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Average pool before the classification.
        filters = opts.nfilters[-1]
        if self.use_lateral:
            self.top_lat_shape = filters
        inshape = np.array([filters, 1, 1])  # Add the shape of the last layer.
        inshapes.append([inshape])
        self.inshapes_devided_by_blocks = inshapes
        self.all_shapes_in_one_list = [item for sublist in inshapes for item in sublist]

    def _make_layer(self, planes: int, nblocks: int, stride: int = 1,num_blocks:int=0) -> nn.Module:
        """
        Args:
            planes: Channels_out of the layer.
            nblocks: num blocks in the layer.
            stride: The stride OF the first block.
            num_blocks: Number of blocks created so far.

        Returns: A ResNet layer.

        """
        layers = []
        layers.append(self.block(self.inplanes, planes, stride, self.use_lateral,num_blocks))  # Add an initial block
        self.inplanes = planes * self.block.expansion
        for idx in range(0, nblocks - 1):  # Add nblocks - 1 preserving blocks.
            layers.append(self.block(self.inplanes, planes, 1, self.use_lateral,num_blocks))
        return layers

class BUModel(nn.Module):
    def __init__(self, opts: argparse,use_embedding:bool):
        """
        Args:
            opts: The model options.
            use_embedding: Whether to access embedding in the stream.
        """

        super(BUModel, self).__init__()
        bu_shared = BUStreamShared(opts)
        self.trunk = BUStream(opts, bu_shared, is_bu2 = use_embedding)  # In the BUModel there is only BU stream.

    def forward(self, inputs: list[torch]) -> list[torch]:
        """
        Args:
            inputs: The BU model inputs.

        Returns: The Model output + lateral connections for the next stream.

        """

        trunk_out, laterals_out = self.trunk(inputs)
        return trunk_out, laterals_out

class BUTDModelShared(nn.Module):
    def __init__(self, opts: argparse):
        """
        Args:
            opts: The model options.
        """
        super(BUTDModelShared, self).__init__()
        self.ntasks = opts.ntasks
        self.use_lateral_butd = opts.use_lateral_butd
        self.use_lateral_tdbu = opts.use_lateral_tdbu
        self.ndirections = opts.ndirections
        self.inputs_to_struct = inputs_to_struct
        self.outs_to_struct = outs_to_struct
        self.lang_embedding = [[] for _ in range(self.ntasks)]  # Container to store the task embedding.
        self.direction_embedding = [[] for _ in range(self.ndirections)]  # Container to store the task embedding.
        self.transfer_learning = [[] for _ in range(self.ntasks)]
        self.argument_embedding = [[] for _ in range(self.ntasks)]
        self.model_flag = opts.model_flag  # The model type
        self.use_bu1_loss = opts.use_bu1_loss  # Whether to use the Occurrence loss.
        self.use_td_flag = opts.use_td_flag
        if self.use_bu1_loss:
            self.occhead = OccurrenceHead(opts)
        bu_shared = BUStreamShared(opts)  # The shared part between BU1, BU2.
        pre_top_shape = bu_shared.inshapes_devided_by_blocks[-2][-1]
        opts.avg_pool_size = tuple(pre_top_shape[1:].tolist())
        self.bu_inshapes = bu_shared.inshapes_devided_by_blocks
        self.bumodel1 = BUStream(opts, bu_shared, is_bu2=False)  # The BU1 stream.
        self.tdmodel = TDModel(opts,bu_shared.inshapes_devided_by_blocks)  # The TD stream.
#        self.use_td_loss = opts.use_td_loss  # Whether to use the TD segmentation loss..
     #   if self.use_td_loss:
    #        self.imagehead = ImageHead(opts)
        self.bumodel2 = BUStream(opts, bu_shared, is_bu2=True)  # The BU2 stream.
        self.Head = MultiTaskHead(opts)  # The task-head to transform the last layer output to the number of classes.
        if self.model_flag is Flag.SF:  # Storing the Task embedding.
            for i in range(self.ntasks):
                self.lang_embedding[i].extend(self.bumodel2.lang_embedding[i])
                self.lang_embedding[i].extend(self.tdmodel.lang_embedding[i])
                self.argument_embedding[i].extend(self.tdmodel.argument_embedding[i])
                self.transfer_learning[i].extend(list(self.Head.taskhead[i].parameters()))
            for i in range(self.ndirections):
                self.direction_embedding[i].extend(self.tdmodel.direction_embedding[i])
                self.direction_embedding[i].extend(self.bumodel2.direction_embedding[i])
        else:
            for i in range(self.ntasks):
                self.task_embedding[i].extend(list(self.Head.taskhead[i].parameters()))

    def forward(self, inputs: list[torch]) -> list[torch]:
        """
        Args:
            inputs: The input from the dataset.

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
        td_outs = self.tdmodel(  model_inputs)  # The input to the TD stream is the bu_out, flags, the lateral connections.
        td_out, td_laterals_out, *td_rest = td_outs
        #TODO - GET RID OF THIS OPTION!
        '''
        if self.use_td_loss:  # Compute the TD head output.
            td_head_out = self.imagehead(td_out)
        '''
        model_inputs = [images, flags]
        if self.use_lateral_tdbu:
            model_inputs += [td_laterals_out]
        else:
            model_inputs += [[td_out]]
        bu2_out, bu2_laterals_out = self.bumodel2(model_inputs)  # The input to the TD stream is the images, flags, the lateral connections.
        head_input = (bu2_out, flags)
        task_out = self.Head(head_input)  # Compute the classification layer.
        outs = [occurrence_out, task_out, bu_out, bu2_out]
        return outs  # Return all the outputs from all streams.

class inputs_to_struct:
    def __init__(self,inputs):
        """
        struct transforming from list of tensors to struct.
        Args:
            inputs: The model inputs including the raw image, the label task, the label all, the label existence and the TD flag.
        """
        img, label_task, label_all, label_existence,char_type_one, task_embd_ohe, arg_emb_ohe, direction_ohe = inputs
        self.image = img
        self.label_all = label_all
        self.label_existence = label_existence
        self.label_task = label_task
        self.flag = [char_type_one,task_embd_ohe,arg_emb_ohe,direction_ohe ]
       # self.task_embd_ohe = task_embd_ohe
      #  self.arg_emb_ohe = arg_emb_ohe
     #   self.direction_ohe = arg_emb_ohe

class outs_to_struct():
    """
    Struct transforming the model output list to struct.
    """
    def __init__(self, outs: list[torch]) -> None:
        """
        # Struct transforming the model output list to struct.
        Args:
            outs: The Model outs containing the BU1 stream out, BU2 stream out and the final output for the classification.
        """

        occurrence_out, task_out, bu_out, bu2_out = outs
        self.occurence_out = occurrence_out
        self.task = task_out
        self.bu = bu_out
        self.bu2 = bu2_out

class BUModelSimple(nn.Module):
    """
    Only BU network.
    """
    def __init__(self, opts):
        """
        Args:
            opts: The model options.
        """
        super(BUModelSimple, self).__init__()
        self.taskhead = MultiTaskHead(opts)
        self.bumodel = BUModel(opts,use_embedding=False)
        pre_top_shape = self.bumodel.trunk.inshapes[-2][1:]
        opts.avg_pool_size = tuple(pre_top_shape[1:].tolist())

    def forward(self, inputs):
        """
        Args:
            inputs: The model input, containing the image and a flag.

        Returns:

        """
        samples = self.inputs_to_struct(inputs)
        images = samples.image
        flags = samples.flag
        model_inputs = [images, flags, None]
        bu_out, _ = self.bumodel(model_inputs)
        task_out = self.taskhead((bu_out,flags))
        return task_out, bu_out

    def outs_to_struct(self, outs):
        """
        Args:
            outs: Model outs containing the task out and the BU2 out.

        Returns: Struct containing the model outs.

        """
        task_out, bu_out = outs
        outs_ns = SimpleNamespace(task=task_out, bu=bu_out)
        return outs_ns

    def inputs_to_struct(self,inputs):
        """
        Struct tasking list of inputs and transform into a struct.
        Args:
            inputs:  image, label_task, flag

        Returns: A struct containing the model inputs.

        """
        image, label_task, flag = inputs
        struct = SimpleNamespace(image = image, label_task = label_task, flag = flag)
        return struct