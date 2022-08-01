import torch.nn as nn
from supp.heads import *
from supp.blocks import *
from supp.general_functions import *
from supp.FlagAt import *
from types import SimpleNamespace
import numpy as np


class TDModel(nn.Module):
    def __init__(self, opts: argparse) -> None:
        """
        :param opts: The options to create the model according to.
        """
        super(TDModel, self).__init__()
        self.block = opts.td_block_type
        self.use_lateral = opts.use_lateral_butd
        self.activation_fun = opts.activation_fun
        self.use_td_flag = opts.use_td_flag
        self.model_flag = opts.model_flag
        self.use_SF = opts.use_SF
        self.orig_relus = opts.orig_relus
        self.norm_layer = opts.norm_fun
        self.top_filters = opts.nfilters[-1]
        self.inplanes = opts.nfilters[-1]
        self.ntasks = opts.ntasks
        self.use_td_flag = opts.use_td_flag
        self.inshapes = opts.bu_inshapes
        self.use_final_conv = opts.use_final_conv
        upsample_size = opts.avg_pool_size  # before avg pool we have 7x7x512
        self.task_embedding = [[] for _ in range(self.ntasks)]
        self.InitialTaskEmbedding = InitialTaskEmbedding(opts)
        for i in range(self.ntasks):
            self.task_embedding[i].extend(self.InitialTaskEmbedding.task_embedding[i])

        self.top_upsample = nn.Upsample(scale_factor=upsample_size, mode='bilinear',
                                        align_corners=False)  # Upsample layer to make at of the shape before the avgpool.
        layers = []
        for k in range(len(opts.strides) - 1, 0, -1):
            nblocks = opts.ns[k]
            stride = opts.strides[k]
            filters = opts.nfilters[k - 1]
            layers.append(self._make_layer(self.block, filters, nblocks, stride=stride, inshapes=self.inshapes,
                                           ntasks=self.ntasks,index = k))  # Create the exact opposite layers of the BU1 stream.
        self.alllayers = nn.ModuleList(layers)
        filters = opts.nfilters[0]
        if self.use_lateral:
            self.bot_lat = SideAndComb(filters, self.norm_layer, self.activation_fun, self.orig_relus, self.ntasks)
        if self.use_final_conv:
            conv1 = conv2d_fun(filters, filters, kernel_size=7, stride=1, padding=3, bias=False)
            self.conv1 = nn.Sequential(conv1, self.norm_layer(filters), self.activation_fun())
        init_module_weights(self.modules())

    def _make_layer(self, block: nn.Module, planes: int, num_blocks: int, stride: int = 1, inshapes: list = None,
                    ntasks: int = 1,index:int=0):
        """
        :param block: Basic TD stream block.
        :param planes: channels_out of the block.
        :param blocks: Number of blocks in the layer.
        :param flag: Model_flag.
        :param stride: The stride.
        :param inshapes: Input shape.
        :param ntasks: Number of tasks the model should handle.
        :return:
        """
        norm_layer = self.norm_layer
        layers = []
        for _ in range(1, num_blocks):  # Create shape preserving blocks.
            newblock = block(self.inplanes, self.inplanes, 1, norm_layer, self.activation_fun, self.use_lateral,
                             inshapes, ntasks,self.orig_relus, index=index)
            layers.append(newblock)
        # Create Upsampling block.
        newblock = block(self.inplanes, planes, stride, norm_layer, self.activation_fun, self.use_lateral, inshapes,
                         ntasks, self.orig_relus,index = index)
        layers.append(newblock)
        self.inplanes = planes * block.expansion
        return nn.ModuleList(layers)

    def forward(self, inputs: list[torch]) -> list[torch]:
        """
        :param inputs:bu_out:the output from the BU1 stream, flag the task+arg flag , laterals_in the laterals from the BU1 stream.
        :return:
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
        if self.use_final_conv:  # Compute the last conv layer.
            x = self.conv1(x)
        laterals_out.append(x)
        outs = [x, laterals_out[::-1]]  # Output the output of the stream + the lateral connections.
        if self.use_td_flag:
            outs += [top_td_embed, top_td]  # Add the top embeddings to the output if needed.
        return outs

class BUStream(nn.Module):
    def __init__(self, opts: argparse, shared: nn.Module, is_bu2: bool) -> None:
        super(BUStream, self).__init__()
        self.block = opts.bu_block_type
        self.inshapes = opts.bu_inshapes
        self.ntasks = opts.ntasks
        self.task_embedding = [[] for _ in range(self.ntasks)]
        self.norm_layer = opts.norm_fun
        self.activation_fun = opts.activation_fun
        self.model_flag = opts.model_flag
        self.inshapes = shared.inshapes_one_list
        self.orig_relus = opts.orig_relus
        self.use_lateral = shared.use_lateral
        self.filters = opts.nfilters[0]
        self.InitialBlock = BUInitialBlock(opts, shared)
        layers = []
        for shared_layer in shared.alllayers:  # For each shared layer we create associate BU layer.
            layers.append(self._make_layer(shared_layer, self.inshapes, is_bu2))
        self.alllayers = nn.ModuleList(layers)
        self.avgpool = shared.avgpool  # Avg pool layer.
        if self.use_lateral:
            self.top_lat = SideAndCombShared(shared.top_lat, self.norm_layer, self.activation_fun, self.orig_relus,
                                             self.ntasks)

        init_module_weights(self.modules())

    def _make_layer(self, blocks: nn.Module, inshapes: list, is_bu2: bool) -> nn.ModuleList:
        """
        :param blocks: Shred layers between BU1, BU2.
        :param inshapes: The input shape of the model.
        :param is_bu2: Whether BU1 or BU2.
        :return: A ResNet shared layer.
        """
        norm_layer = self.norm_layer
        layers = []
        for shared_block in blocks:
            # Create Basic BU block.
            layer = self.block(shared_block, norm_layer, self.activation_fun, inshapes, self.ntasks, self.model_flag,
                               is_bu2, self.orig_relus)
            if self.model_flag is FlagAt.SF and is_bu2:
                # Adding the task embedding of the BU2 stream.
                for i in range(self.ntasks):
                    self.task_embedding[i].extend(layer.task_embedding[i])
            layers.append(layer)
        return nn.ModuleList(layers)

    def forward(self, inputs: list[torch]) -> tuple:
        x, flags, laterals_in = inputs  # The input is the image, the flag and the lateral from the previous stream.
        laterals_out = []  # The laterals for the second stream.
        x = self.InitialBlock((x, flags, laterals_in))  # Compute the initial block in ResNet.
        laterals_out.append(x)
        for layer_id, layer in enumerate(self.alllayers):
            layer_lats_out = []
            for block_id, block in enumerate(layer):
                lateral_layer_id = layer_id + 1
                cur_lat_in = get_laterals(laterals_in, lateral_layer_id,
                                          block_id)  # Get the laterals associate with the layer,block_id.
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
    def __init__(self, opts: argparse) -> None:
        """
        :param opts: The parameters to create the model according to.
        """
        super(BUStreamShared, self).__init__()
        block_id = 0
        layers = []
        self.activation_fun = opts.activation_fun
        self.use_lateral = opts.use_lateral_tdbu
        self.use_bu1_flag = opts.use_bu1_flag
        self.block = opts.bu_shared_block_type
        stride = opts.strides[0]
        filters = opts.nfilters[0]
        inplanes = opts.inshape[0]
        inshape = np.array(opts.inshape)
        inshapes = []  # The shapes of all tensors in all blocks.
        inshapes_one_list = []
        self.conv1 = depthwise_separable_conv(inplanes, filters, kernel_size=7, stride=stride, padding=3,
                                              bias=False)  # The first conv layer as in ResNet.
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
        :param planes: Channels_out of the layer.
        :param nblocks: num blocks in the layer.
        :param stride:  The stride
        :param num_blocks: Num blocks created so far.
        :return:
        """
        layers = []
        layers.append(self.block(self.inplanes, planes, stride, self.use_lateral,num_blocks))  # Add an initial block
        self.inplanes = planes * self.block.expansion
        for idx in range(0, nblocks - 1):  # Add nblocks - 1 preserving blocks.
            layers.append(self.block(self.inplanes, planes, 1, self.use_lateral,num_blocks))
        return layers


class BUModel(nn.Module):
    def __init__(self, opts: argparse) -> None:
        """
        :param opts: opts to create the model according to.
        """
        super(BUModel, self).__init__()
        bu_shared = BUStreamShared(opts)
        self.trunk = BUStream(opts, bu_shared, is_bu2=False)  # In the BUModel there is only BU stream.

    def forward(self, inputs: list[torch]) -> tuple:
        """
        :param inputs: Only the images.
        :return: The output + lateral connections.
        """
        trunk_out, laterals_out = self.trunk(inputs)
        return trunk_out, laterals_out


class BUTDModel(nn.Module):
    def forward(self, samples: list[torch], stage:int ) -> list[torch]:
        """
        :param inputs: The images, all labels including the task_occurrence, segmentation_task, the label_task.
        :return: The output from all streams in order to compute the appropriate loss.
        """
      #  samples = self.inputs_to_struct(inputs)  # Transform the input to struct.
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
        if self.use_td_loss:  # Compute the TD head output.
            td_head_out = self.imagehead(td_out)
        model_inputs = [images, flags]
        if self.use_lateral_tdbu:
            model_inputs += [td_laterals_out]
        else:
            model_inputs += [[td_out]]
        bu2_out, bu2_laterals_out = self.bumodel2(
            model_inputs)  # The input to the TD stream is the images, flags, the lateral connections.
        head_input = (bu2_out, flags,stage)
        task_out = self.Head(head_input)  # Compute the classification layer.
        outs = [occurrence_out, task_out, bu_out, bu2_out]
        if self.use_td_loss:
            outs += [td_head_out]
        if self.tdmodel.use_td_flag:
            td_top_embed, td_top = td_rest
            outs += [td_top_embed]
        return outs  # Return all the outputs from all streams.

    class outs_to_struct:
        """
        Struct transforming the model output list to struct.
        """

        def __init__(self, model: nn.Module) -> None:
            """
            :param model:Containing the flags to create the model according to.
            """
            self.use_td_loss = model.use_td_loss
            self.use_td_flag = model.use_td_flag
            self.occurrence_out = None
            self.task_out = None
            self.bu_out = None
            self.bu2_out = None
            self.td_head_out = None
            self.td_top_embe = None

        def __call__(self, outs: list[torch]) -> object:
            """
            :param outs: List of all model streams output.
            :return: Struct instance.
            """
            occurrence_out, task_out, bu_out, bu2_out, *rest = outs
            self.occurrence_out = occurrence_out
            self.task = task_out
            self.bu = bu_out
            self.bu2 = bu2_out
            if self.use_td_loss:
                td_head_out, *rest = rest
                self.td_head = td_head_out
            if self.use_td_flag:
                td_top_embed = rest[0]
                self.td_top_embed = td_top_embed
            return self


class BUTDModelShared(BUTDModel):
    def __init__(self, opts: argparse) -> None:
        """
        :param opts: opts to create the model according to.
        """
        super(BUTDModelShared, self).__init__()
        self.ntasks = opts.ntasks
        self.use_lateral_butd = opts.use_lateral_butd
        self.use_lateral_tdbu = opts.use_lateral_tdbu
        self.inputs_to_struct = opts.inputs_to_struct
        self.task_embedding = [[] for _ in range(self.ntasks)]  # Container to store the task embedding.
        self.transfer_learning = [[] for _ in range(self.ntasks)]
        self.model_flag = opts.model_flag  # The model type
        self.use_bu1_loss = opts.use_bu1_loss  # Whether to use the Occurrence loss.
        self.use_td_flag = opts.use_td_flag
        if self.use_bu1_loss:
            self.occhead = OccurrenceHead(opts)
        bu_shared = BUStreamShared(opts)  # The shared part between BU1, BU2.
        pre_top_shape = bu_shared.inshapes[-2][-1]
        opts.avg_pool_size = tuple(pre_top_shape[1:].tolist())
        opts.bu_inshapes = bu_shared.inshapes
        self.bumodel1 = BUStream(opts, bu_shared, is_bu2=False)  # The BU1 stream.
        self.tdmodel = TDModel(opts)  # The TD stream.
        self.use_td_loss = opts.use_td_loss  # Whether to use the TD segmentation loss..
        if self.use_td_loss:
            self.imagehead = ImageHead(opts)
        self.bumodel2 = BUStream(opts, bu_shared, is_bu2=True)  # The BU2 stream.
        self.Head = MultiTaskHead(opts)  # The task-head to transform the last layer output to the number of classes.
        if self.model_flag is FlagAt.SF:  # Storing the Task embedding.
            for i in range(self.ntasks):
                self.task_embedding[i].extend(list(self.Head.taskhead[i].parameters()))
                self.task_embedding[i].extend(self.bumodel2.task_embedding[i])
                self.task_embedding[i].extend(self.tdmodel.task_embedding[i])
                self.transfer_learning[i].extend(list(self.Head.taskhead[i].parameters()))
                self.transfer_learning[i].extend(self.tdmodel.task_embedding[i])
        else:
            for i in range(self.ntasks):
                self.task_embedding[i].extend(list(self.Head.taskhead[i].parameters()))

class cyclic_inputs_to_strcut:
    def __init__(self,inputs,stage):
        img, label_all, label_existence, flag_stage_1, flag_stage_2, flag_stage_3 ,label_task_stage_1, label_task_stage_2, label_task_stage_3  = inputs
        self.img = img
        self.label_all = label_all
        self.label_existence = label_existence
        if stage == 0:
         self.label_task = label_task_stage_1
         self.flag = flag_stage_1
        if stage == 1:
         self.label_task = label_task_stage_2
         self.flag = flag_stage_2
        if stage == 2:
         self.label_task = label_task_stage_3
         self.flag = flag_stage_3

class CYCLICBUTDMODELSHARED(nn.Module):
    def __init__(self,opts):
        self.model = BUTDModelShared(opts)

    def forward(self, inputs: list[torch]) -> list[torch]:
        samples_stage_1 = inputs_to_struct(inputs,stage = 0)
        out_stage_1 = self.model(samples_stage_1,stage = 0 )
        #
        samples_stage_2 = inputs_to_struct(inputs, stage=1)
        out_stage_2 = self.model(samples_stage_2,stage = 1)
        #
        samples_stage_3 = inputs_to_struct(inputs, stage=2)
        out_stage_3 = self.model(samples_stage_3,stage = 2)
        #
        return [out_stage_1, out_stage_2, out_stage_3]

class BUTDModelDuplicate(BUTDModel):
    """
    The same as BUTDModelShared but model1 == model2.
    """

    def __init__(self, opts):
        super(BUTDModelDuplicate, self).__init__()
        self.occhead = OccurrenceHead(opts)
        self.taskhead = MultiLabelHead(opts)
        self.use_td_loss = opts.use_td_loss
        if self.use_td_loss:
            self.imagehead = ImageHead(opts)
        bu_shared = BUStreamShared(opts)
        self.bumodel1 = BUStream(opts, bu_shared, is_bu2=False)
        opts.use_top_flag = opts.use_bu2_flag
        self.bumodel2 = self.bumodel1
        pre_top_shape = bu_shared.inshapes[-2][-1]
        opts.avg_pool_size = tuple(pre_top_shape[1:].tolist())
        self.tdmodel = TDModel(opts)
        self.use_bu1_flag = opts.use_bu1_flag
        self.use_lateral_butd = opts.use_lateral_butd
        self.use_lateral_tdbu = opts.use_lateral_tdbu
        self.inputs_to_struct = v


class BUTDModelSeparate(BUTDModel):
    """
    The same as BUTDShared in the architecture but model1, model2 are not constrained to share the same weights.
    """
    def __init__(self, opts):
        super(BUTDModelSeparate, self).__init__()
        self.occhead = OccurrenceHead(opts)
        self.taskhead = MultiLabelHead(opts)
        self.use_td_loss = opts.use_td_loss
        if self.use_td_loss:
            self.imagehead = ImageHead(opts)
        bu_shared1 = BUStreamShared(opts)
        self.bumodel1 = BUStream(opts, bu_shared1, is_bu2=False)
        opts.use_top_flag = opts.use_bu2_flag
        bu_shared2 = BUStreamShared(opts)
        self.bumodel2 = BUStream(opts, bu_shared2, is_bu2=True)
        pre_top_shape = bu_shared1.inshapes[-2][-1]
        opts.avg_pool_size = tuple(pre_top_shape[1:].tolist())
        self.tdmodel = TDModel(opts)
        self.use_bu1_flag = opts.use_bu1_flag
        self.use_lateral_butd = opts.use_lateral_butd
        self.use_lateral_tdbu = opts.use_lateral_tdbu
        self.inputs_to_struct = opts.inputs_to_struct


class BUModelSimple(nn.Module):
    """
    Only BU network.
    """

    def __init__(self, opts):
        super(BUModelSimple, self).__init__()
        self.occhead = OccurrenceHead(opts)
        self.taskhead = MultiLabelHead(opts)
        self.bumodel = BUModel(opts)
        pre_top_shape = self.bumodel.trunk.inshapes[-2][-1]
        opts.avg_pool_size = tuple(pre_top_shape[1:].tolist())
        self.inputs_to_struct = opts.inputs_to_struct

    def forward(self, inputs):
        samples = self.inputs_to_struct(inputs)
        images = samples.image
        flags = samples.flag
        model_inputs = [images, flags, None]
        bu_out, _ = self.bumodel(model_inputs)
        occurrence_out = self.occhead(bu_out)
        task_out = self.taskhead(bu_out)
        return occurrence_out, task_out, bu_out

    def outs_to_struct(self, outs):
        occurrence_out, task_out, bu_out = outs
        outs_ns = SimpleNamespace(occurrence=occurrence_out, task=task_out, bu=bu_out)
        return outs_ns
