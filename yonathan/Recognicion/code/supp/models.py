import argparse

import numpy as np
import torch
import torch.nn as nn

from supp.Dataset_and_model_type_specification import Flag, DsType
from supp.Module_Blocks import  SideAndComb, init_module_weights
from supp.Model_Blocks import BUInitialBlock, InitialTaskEmbedding
from supp.Module_Blocks import Depthwise_separable_conv
from supp.heads import MultiTaskHead, OccurrenceHead
from supp.utils import get_laterals
from typing import Union

class BUStreamShared(nn.Module):
    def __init__(self, opts: argparse):
        """
        The Shared stream between BU1, BU2.
        Args:
            opts: Model options.
        """
        super(BUStreamShared, self).__init__()
        self.use_lateral = opts.use_lateral_tdbu
        self.block = opts.bu_shared_block_type
        self.inshapes = opts.inshape[0]
        self.opts = opts
        self.inplanes = opts.nfilters[0]
        self.nfilters_bot_lat = self.inplanes
        self.alllayers = []
        stride = opts.strides[0]
        inshape = np.array(opts.inshape)
        self.conv1 = Depthwise_separable_conv(self.inshapes, self.inplanes, kernel_size=7, stride=stride, padding=3,
                                              bias=False)  # The first conv layer as in ResNet.
        inshape = np.array( [self.inplanes, np.int(np.ceil(inshape[1] / stride)), np.int(np.ceil(inshape[2] / stride))])  # The first shape.
        inshapes = [inshape]
        for k in range(1, len(opts.strides)): # For each stride create layer of Blocks.
            nblocks = opts.ns[k] # The number of blocks in the layer.
            stride = opts.strides[k] # The stride.
            filters = opts.nfilters[k] # The number of filters transform to.
            self.alllayers.append(self._make_layer(filters, nblocks, stride=stride)) # Making a layer.
            inshape = np.array([filters, np.int(np.ceil(inshape[1] / stride)), np.int(np.ceil(inshape[2] / stride))]) # Compute the output shape of the layer.
            inshapes.append(inshape) # Add the shape to the shapes list.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Average pool before the classification.
        if self.use_lateral:
            self.nfilters_top_lat = opts.nfilters[-1] # The number of filters for the lateral connection.
        self.inshapes = inshapes

    def _make_layer(self, planes: int, nblocks: int, stride: int) -> nn.Module:
        """
        Args:
            planes: Channels_out of the layer.
            nblocks: num blocks in the layer.
            stride: The stride.

        Returns: A ResNet layer.

        """
        layers = [] # All Blocks list.
        layers.append(self.block(self.opts, self.inplanes, planes, stride))  # Add an initial block
        self.inplanes = planes * self.block.expansion
        for idx in range(0, nblocks - 1):  # Add nblocks - 1 preserving blocks.
            layers.append(self.block(self.opts, self.inplanes, planes, 1))
        return layers

class BUModel(nn.Module):
    def __init__(self, opts:int, use_task_embedding:int):
        """
        Args:
            opts:
            use_task_embedding:
        """
        super(BUModel,self).__init__()
        bu_shared = BUStreamShared(opts = opts)
        self.model = BUStream(opts = opts, shared = bu_shared, is_bu2 = use_task_embedding)

    def forward(self,inputs):
        """
        Args:
            inputs:

        Returns:

        """
        return self.model(inputs)

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
        self.ndirections = opts.ndirections
        self.task_embedding = [[] for _ in range(self.ndirections)]
        self.model_flag = opts.model_flag
        self.inshapes = shared.inshapes
        self.opts = opts
        self.use_lateral = shared.use_lateral
        self.InitialBlock = BUInitialBlock(opts, shared)
        layers = []
        for layer_idx, shared_layer in enumerate(shared.alllayers):  # For each shared layer we create associate BU layer.
            layers.append(self._make_layer(shared_layer, is_bu2, layer_idx))
        self.alllayers = nn.ModuleList(layers)
        self.avgpool = shared.avgpool  # Avg pool layer.
        use_lateral = shared.use_lateral
        if use_lateral:
            self.top_lat = SideAndComb(opts, shared.nfilters_top_lat)
        ndirections = opts.ndirections
        self.task_embedding = [[] for _ in range(ndirections)]

        init_module_weights(self.modules())

    def _make_layer(self, blocks: nn.Module, is_bu2: bool, layer_id: int) -> nn.ModuleList:
        """
        Args:
            blocks: Shared layers between BU1, BU2.
            inshapes: The input shape of the model.
            is_bu2: Whether BU1 or BU2.

        Returns: A ResNet shared layer.

        """
        layers = []
        for shared_block in blocks:
            # Create Basic BU blocks for each shared block.
            block_inshape = self.inshapes[layer_id + 1]
            layer = self.block(self.opts, shared_block, block_inshape, is_bu2, self.task_embedding)
            layers.append(layer)
        return nn.ModuleList(layers)

    def forward(self, inputs: list[torch]) -> tuple:
        """
        Args:
            inputs: The input, the flags, the lateral connection from TD network.

        Returns: The model output + The lateral connections.

        """
        x, flags, laterals_in = inputs  # The input is the image, the flag and the lateral from the previous stream(if exist).
        laterals_out = []  # The laterals for the second stream.
        x = self.InitialBlock((x, flags, laterals_in))  # Compute the initial block in ResNet.
        laterals_out.append([x])
        for layer_id, layer in enumerate(self.alllayers):
            layer_lats_out = []
            for block_id, block in enumerate(layer):
                lateral_layer_id = layer_id + 1 # Get layer id.
                cur_lat_in = get_laterals(laterals_in, lateral_layer_id, block_id)  # Get the laterals associate with the layer, block_id if exist.
                x, block_lats_out = block((x, flags, cur_lat_in))  # Compute the block with the lateral connection.
                layer_lats_out.append(block_lats_out)
            laterals_out.append(layer_lats_out)
        x = self.avgpool(x)  # Avg pool.
        lateral_in = get_laterals(laterals_in, lateral_layer_id + 1, 0) # The last lateral connection.
        if self.use_lateral and lateral_in is not None:
            x = self.top_lat((x, lateral_in))  # last lateral connection before the the loss.
        laterals_out.append([x])
        return x, laterals_out

class TDModel(nn.Module):
    def __init__(self, opts: argparse, bu_inshapes: list):
        """
        Args:
            opts: The model opts.
            bu_inshapes: The BU shapes.
        """
        super(TDModel, self).__init__()
        self.block = opts.td_block_type
        self.use_lateral = opts.use_lateral_butd
        self.model_flag = opts.model_flag
        self.inplanes = opts.nfilters[-1]
        self.opts = opts
        self.ntasks = opts.ntasks
        self.inshapes = bu_inshapes
        self.ndirections = opts.ndirections
        upsample_size = opts.avg_pool_size  # before avg pool we have 7x7x512
        self.task_embedding = [[] for _ in range(self.ndirections)]
        self.argument_embedding = [[] for _ in range(self.ntasks)]
        self.InitialTaskEmbedding = InitialTaskEmbedding(opts,self.task_embedding) # Initial TD block.
        if opts.ds_type is DsType.Omniglot and self.model_flag is Flag.CL:
            for j in range(self.ntasks):
                self.argument_embedding[j].extend(self.InitialTaskEmbedding.argument_embedding[j])
        self.top_upsample = nn.Upsample(scale_factor=upsample_size, mode='bilinear',
                                        align_corners=False)  # Upsample layer to make at of the shape before the avgpool.
        self.alllayers = nn.ModuleList()
        for k in range(len(opts.strides) - 1, 0, -1):
            nblocks = opts.ns[k]
            stride = opts.strides[k]
            filters = opts.nfilters[k - 1]
            self.alllayers.append(self._make_layer(filters, nblocks, stride=stride, index = k))  # Create the exact opposite layers of the BU1 stream.
        if self.use_lateral:
            self.bot_lat = SideAndComb(opts, opts.nfilters[0])
        init_module_weights(self.modules())

    def _make_layer(self, planes: int, num_blocks: int, stride: int = 1, index: int = 0):
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
            block_inshape = self.inshapes[index - 1]
            newblock = self.block(self.opts, self.inplanes, self.inplanes, 1, block_inshape)
            layers.append(newblock)

        block_inshape = self.inshapes[index - 1] #Compute the shape upsample to.
        newblock = self.block(self.opts, self.inplanes, planes, stride, block_inshape) # Create Upsampling block.
        layers.append(newblock)
        self.inplanes = planes * self.block.expansion
        return nn.ModuleList(layers)

    def forward(self, inputs: tuple[torch]) -> tuple[torch]:
        """
        Args:
            inputs: The output from the BU1 stream, flag the task+arg flag , laterals_in, the laterals from the BU1 stream.

        Returns: The td outputs + lateral connections foR bu1.

        """
        bu_out, flag, laterals_in = inputs
        laterals_out = []
        if self.model_flag is not Flag.NOFLAG:
            x = self.InitialTaskEmbedding((bu_out, flag))  # Compute the initial task embedding.
        else:
            x = bu_out
        laterals_out.append([x])
        x = self.top_upsample(x)  # Upsample to the shape before the avgpooling in the BU1 stream.
        use_lateral = not(laterals_in is None or not self.use_lateral)
        # If we have lateral connections and use them we invert them to match our size else None
        reverse_laterals_in = laterals_in[::-1] if use_lateral else None
        for layer_id, layer in enumerate(self.alllayers):  # Iterating over all layers in the stream.
            layer_lats_out = []  # The lateral connections for the BU2 stream.
            for block_id, block in enumerate(layer):  # Iterating over all blocks in the layer.
                try: # if we have None we except.
                  reverse_cur_lat_in = get_laterals(reverse_laterals_in,layer_id + 1, block_id)[::-1]  # Inverting the laterals to match the desired shape.
                except:
                    reverse_cur_lat_in = None
                x, block_lats_out = block((x, flag, reverse_cur_lat_in))  # Compute the block output using x, the flag and the lateral connections.
                layer_lats_out.append(block_lats_out)  # Add the lateral output for the next stream.
            reverse_layer_lats_out = layer_lats_out[::-1]
            laterals_out.append(reverse_layer_lats_out)  # Add all the layer's laterals.
        lateral_in = reverse_laterals_in[-1][0] if use_lateral else None
        if self.use_lateral:
            x = self.bot_lat((x, lateral_in))  # Compute lateral connection + channel modulation.
        laterals_out.append([x])
        outs = [x, laterals_out[::-1]]  # Output the output of the stream + the lateral connections.
        return outs

class BUTDModelShared(nn.Module):
    def __init__(self, opts: argparse):
        """
        The main model.
        The full BU-TD model with possible embedding for continual learning.
        Args:
            opts: Model options.
        """
        super(BUTDModelShared, self).__init__()

        self.task_embedding = None
        self.argument_embedding = None
        self.use_lateral_butd = opts.use_lateral_butd
        self.use_lateral_tdbu = opts.use_lateral_tdbu
        self.model_flag = opts.model_flag  # The model type
        self.use_bu1_loss = opts.use_bu1_loss  # Whether to use the Occurrence loss.
        self.ds_type = opts.ds_type
        if self.use_bu1_loss:
            self.occhead = OccurrenceHead(opts)
        bu_shared = BUStreamShared(opts)  # The shared part between BU1, BU2.
        shapes = bu_shared.inshapes
        opts.avg_pool_size = tuple(shapes[-1][1:])
        self.bu_inshapes = bu_shared.inshapes
        self.bumodel1 = BUStream(opts, bu_shared, is_bu2=False)  # The BU1 stream.
        self.tdmodel = TDModel(opts, shapes)  # The TD stream.
        self.bumodel2 = BUStream(opts, bu_shared, is_bu2=True)  # The BU2 stream.
        ntasks = opts.ntasks
        ndirections = opts.ndirections
        self.transfer_learning = [[] for _ in range(ndirections * ntasks)]
        self.Head = MultiTaskHead(opts, self.transfer_learning)  # The task-head to transform the last layer output to the number of classes.
        if self.model_flag is Flag.CL:  # Storing the Task embedding.
            self.task_embedding = list(map(list.__add__,self.bumodel2.task_embedding, self.tdmodel.task_embedding))
            if self.ds_type is DsType.Omniglot:
                self.argument_embedding = self.tdmodel.argument_embedding

    def forward(self, inputs: list[torch]) -> list[torch]:
        """
        Args:
            inputs: The model input.

        Returns: The output from all streams.

        """
        samples = self.inputs_to_struct(inputs)  # Making the input to a struct.
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
        td_outs = self.tdmodel(model_inputs)  # The input to the TD stream is the bu_out, flags, the lateral connections.
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

    class inputs_to_struct:
        # class receiving list of tensors and makes to a class.
        def __init__(self, inputs:tuple[torch]):
            """
            Args:
                inputs: The tensor list.
            """
            img, label_task, flag, label_all, label_existence = inputs
            self.image = img
            self.label_all = label_all
            self.label_existence = label_existence
            self.label_task = label_task
            self.flag = flag

    class outs_to_struct:
        def __init__(self, outs: list[torch]):
            """
            Struct transforming the model output list to struct.
            Args:
                outs: The model outs.
            """
            occurrence_out, task_out, bu_out, bu2_out = outs
            self.occurrence_out = occurrence_out
            self.classifier = task_out
            self.bu = bu_out
            self.bu2 = bu2_out

class ResNet(nn.Module):
    """
    A ResNet model.
    Usually used as a Baseline model comapring to BU-TD in Continual learning.
    """
    def __init__(self, opts):
        """
        Args:
            opts:  The model options.
        """
        super(ResNet, self).__init__()

        self.opts = opts
        self.ndirections = opts.ndirections
        self.feauture = BUModel(opts, use_task_embedding = False)
        self.classifier = MultiTaskHead(opts)

    def forward_features(self, features, inputs, head = None):
        """
        Needed for switching a task-head given computed features.
        Args:
            features: The computed features.
            inputs: The input.
            head: The head we want to switch to if head != None.

        Returns: The switched taskout.

        """
        samples = self.opts.inputs_to_struct(inputs)
        flags = samples.flag # Getting the flags.
        task_out = self.taskhead((features, flags), head)  # Compute the switched taskhead.
        return task_out

    def forward(self, inputs,head = None):
        """
        Computing the forward of the head to get class scores.
        Args:
            inputs: The inputs to the model.
            head: A head id if we want to switch to.

        Returns: The features and class scores.

        """
        samples = self.inputs_to_struct(inputs)
        images = samples.image
        flags = samples.flag
        model_inputs = [images, flags, None]
        features, _ = self.feauture(model_inputs)
        classifier = self.classifier((features, flags),idx_out = head)
        return  features, classifier

    class inputs_to_struct:
        # class receiving list of tensors and makes to a class.
        def __init__(self, inputs):
            """
            Args:
                inputs: The tensor list.
            """
            img, label_task, flag, label_all, label_existence = inputs
            self.image = img
            self.label_all = label_all
            self.label_existence = label_existence
            self.label_task = label_task
            self.flag = flag

    def get_features(self, inputs:tuple[torch])->torch:
        """
        Compute only the features of the model.
        Args:
            inputs: The model inputs.

        Returns: The features.

        """
        samples = self.opts.inputs_to_struct(inputs) # Get the samples.
        images = samples.image # The image.
        flags = samples.flag # The flag.
        model_inputs = [images, flags, None]
        features, _ = self.bumodel(model_inputs) # The model output.
        return features

    class outs_to_struct:
        def __init__(self, outs: list[torch]):
            """
            Struct transforming the model output list to struct.
            Args:
                outs: The model outs.
            """
            feautures, classifier = outs
            self.features = feautures
            self.classifier = classifier

    def forward_and_out_to_struct(self,inputs:tuple[torch], head:Union[int,None] = None):
        """
        Args:
            inputs: The model input.
            head: The head if we desire to switch to.

        Returns: A struct containing the features and the class scores.

        """
        outs = self(inputs, head)
        return self.outs_to_struct(outs)

