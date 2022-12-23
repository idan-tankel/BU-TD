import argparse


from typing import Iterator

import numpy as np
import torch
import torch.nn as nn

from training.Data.Data_params import Flag, DsType
from training.Data.Structs import inputs_to_struct
from training.Modules.Heads import MultiTaskHead, OccurrenceHead
from training.Modules.Model_Blocks import BUInitialBlock, init_module_weights, InitialEmbeddingBlock
from training.Modules.Module_Blocks import Depthwise_separable_conv, Modulation_and_Lat
from training.Utils import get_laterals, tuple_direction_to_index


# Here we define the Models.
# It includes BUTDModel, BUModel and Pure ResNet.

class BUStreamShared(nn.Module):
    # Here we define the shared part between BU1, BU2.
    def __init__(self, opts: argparse):
        """
        The Shared stream between BU1, BU2, contain only conv layers.
        Args:
            opts: Model options.
        """
        super(BUStreamShared, self).__init__()
        self.opts = opts  # The model options.
        self.block = opts.bu_shared_block_type  # The block type.
        Model_inshape = np.array(opts.inshape)  # The image resolution.
        self.conv1 = Depthwise_separable_conv(Model_inshape[0], opts.nfilters[0], kernel_size=7, stride=opts.strides[0],
                                              padding=3,
                                              bias=False)  # The first conv layer as in ResNet.

        # The first shape after Conv1.
        inshape = np.array([opts.nfilters[0], np.int(np.ceil(Model_inshape[1] / opts.strides[0])),
                            np.int(np.ceil(Model_inshape[2] / opts.strides[0]))])
        self.inshapes = [inshape]  # list that should contain all layer output shapes.
        self.alllayers = []  # All layers list.
        inplanes = opts.nfilters[0]
        for k in range(len(opts.strides)):  # For each stride create layer of Blocks.
            nblocks = opts.ns[k]  # The number of blocks in the layer.
            stride = opts.strides[k]  # The stride.
            filters = opts.nfilters[k]  # The number of filters transform to.
            if nblocks > 0:
                self.alllayers.append(self._make_layer(inplanes, filters, nblocks, stride=stride))  # Making a layer.
                # Compute the output shape of the layer.
                inshape = np.array([filters, np.int(np.ceil(inshape[1] / stride)),
                                    np.int(np.ceil(inshape[2] / stride))])
                self.inshapes.append(inshape)  # Add the shape to the shapes list.
                inplanes = filters  # Update the future inplanes to be the current filters.

    def _make_layer(self, inplanes: int, planes: int, nblocks: int, stride: int) -> list[nn.Module]:
        """
        Args:
            inplanes: Channels_in of the layer.
            planes: Channels_out of the layer.
            nblocks: num blocks in the layer.
            stride: The stride.

        Returns: A ResNet layer.
        """
        planes = planes * self.block.expansion
        layers = [self.block(self.opts, inplanes, planes, stride)]  # Create a block.
        for idx in range(0, nblocks - 1):  # Add nblocks - 1 preserving blocks.
            layers.append(self.block(self.opts, planes, planes, 1))
        return layers


class BUModel(nn.Module):
    # Equivalent to Pure ResNet.
    def __init__(self, opts: argparse, use_task_embedding: bool = False):
        """
        Args:
            opts: The model opts.
            use_task_embedding: Whether to create the task embedding.
        """
        super(BUModel, self).__init__()
        bu_shared = BUStreamShared(opts=opts)  # The shared model part.
        self.model = BUStream(opts=opts, shared=bu_shared, is_bu2=use_task_embedding)  # Create the model.

    def forward(self, inputs: list[torch]) -> list[torch]:
        """
        Args:
            inputs: The model input.

        Returns: The model output.

        """
        return self.model(inputs)


class BUStream(nn.Module):

    def __init__(self, opts: argparse, shared: nn.Module, is_bu2: bool = False):
        """
        Args:
            opts: The model options.
            shared: The shared part between BU1, BU2.
            is_bu2: Whether the stream is BU1 or BU2.
        """
        super(BUStream, self).__init__()
        self.opts = opts  # Save the model opts.
        self.block = opts.bu_block_type  # The basic block type.
        self.task_embedding = [[] for _ in range(opts.ndirections)]  # List should contain the task embedding.
        self.inshapes = shared.inshapes  # The output shape of all layers.
        self.use_lateral = opts.use_lateral_tdbu  # Whether to use the TD -> BU2 laterals.
        self.is_bu2 = is_bu2  # Save whether we are on the BU2 stream.
        self.InitialBlock = BUInitialBlock(opts, shared,
                                           is_bu2=is_bu2,task_embedding = self.task_embedding)  # The initial block, getting the image as an input.
        self.alllayers = nn.ModuleList()
        # For each shared layer we create associate BU layer.
        for layer_idx, shared_layer in enumerate(shared.alllayers):
            self.alllayers.append(self._make_layer(shared_layer, layer_idx))
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # Avg pool layer.
        # Last lateral connection fot the beginning of the TD-stream.
        if self.use_lateral and is_bu2:
            self.top_lat = Modulation_and_Lat(opts, opts.nfilters[-1])
        init_module_weights(self.modules())  # Initialize the weights.

    def _make_layer(self, blocks: Iterator[nn.Module], layer_id: int) -> nn.ModuleList:
        """
        Args:
            blocks: Shared conv layers between BU1, BU2.
            layer_id: The layer id for the input shape computing.

        Returns: A ResNet shared layer.

        """
        layers = nn.ModuleList()
        for shared_block in blocks:
            # Create Basic BU blocks for each shared block.
            block_inshape = self.inshapes[layer_id + 1]  # The block input shape, needed for the task embedding.
            layer = self.block(self.opts, shared_block, block_inshape, self.is_bu2, self.task_embedding)
            layers.append(layer)  # Add the layer.
        return layers

    def forward(self, inputs: list[torch]) -> tuple:
        """
        The forward includes processing the input and inserting the lateral connections.
        Args:
            inputs: The input, the flags, the lateral connection from TD network.

        Returns: The model output + The lateral connections.

        """
        # The input is the image, the flag and the lateral connections from the previous stream(if exist).
        x, flags, laterals_in = inputs
        laterals_out = []  # The laterals for the second stream.
        x = self.InitialBlock(x,flags, laterals_in)  # Compute the initial block in ResNet.
        laterals_out.append([x])
        for layer_id, layer in enumerate(self.alllayers):
            layer_lats_out = []  # The lateral connections for the next stream.
            for block_id, block in enumerate(layer):
                lateral_layer_id = layer_id + 1  # Get layer id.
                cur_lat_in = get_laterals(laterals_in, lateral_layer_id,
                                          block_id)  # Get the laterals associate with the layer, block_id(if exist).
                x, block_lats_out = block(x, flags, cur_lat_in)  # Compute the block with the lateral connection.
                layer_lats_out.append(block_lats_out)
            laterals_out.append(layer_lats_out)
        x = self.avgpool(x)  # Avg pool.
        lateral_in = get_laterals(laterals_in, len(self.alllayers) + 1, 0)  # The last lateral connection.
        if self.use_lateral and lateral_in is not None:
            x = self.top_lat(x, lateral_in)  # last lateral connection before the end.
        laterals_out.append([x])  # Add to laterals.
        x = x.squeeze()  # Squeeze to be a vector.
        return x, laterals_out


class TDModel(nn.Module):
    def __init__(self, opts: argparse, bu_inshapes: list):
        """
        Args:
            opts: The model opts.
            bu_inshapes: The BU shapes.
        """
        super(TDModel, self).__init__()
        self.opts = opts  # Save the opts.
        self.block = opts.td_block_type  # The block type
        self.use_lateral = opts.use_lateral_butd  # Whether to use the BU1 -> TD laterals.
        self.ntasks = opts.ntasks  # The number of tasks
        self.inshapes = bu_inshapes  # The model layers output.
        self.argument_embedding = [[] for _ in range(self.ntasks)]
        # Only in Omniglot we have several tasks, with different argument embeddings.
        self.use_initial_emb = opts.model_flag is not Flag.NOFLAG
        if self.use_initial_emb:
            # Create the initial TD block.
            self.InitialTaskEmbedding = InitialEmbeddingBlock(opts)
        # Upsample layer to match the shape before the avgpool.
        shape_before_avg_pool = tuple(bu_inshapes[-1][1:])  # The last shape of BU1, we task the inner shape.
        inshape = bu_inshapes[-1][0]
        self.top_upsample = nn.Upsample(scale_factor=shape_before_avg_pool, mode='bilinear',
                                        align_corners=False)
        # List contain all layers.
        self.alllayers = nn.ModuleList()
        # Create layers.
        for k in range(len(opts.strides) - 1, 0, -1):
            nblocks = opts.ns[k]  # The number of blocks in the layer.
            stride = opts.strides[k]  # The current stride.
            filters = opts.nfilters[k - 1]  # The number of filters.
            if nblocks > 0:
                self.alllayers.append(self._make_layer(inshape, filters, nblocks, stride=stride,
                                                       index=k))  # Create the exact opposite layers of the BU1 stream.
            inshape = filters  # The current output filters, is the next layer input filter.
        # The last lateral connection
        if self.use_lateral:
            self.bot_lat = Modulation_and_Lat(opts, opts.nfilters[0])
        # Copy the argument embedding.
        if opts.model_flag is Flag.CL:
            for j in range(self.ntasks):
                self.argument_embedding[j].extend(self.InitialTaskEmbedding.top_td_arg_emb[j].parameters())
        init_module_weights(self.modules())  # Initialize the weights.

    def _make_layer(self, inplanes: int, planes: int, num_blocks: int, stride: int = 1, index: int = 0):
        """
        Args:
            inplanes: The input filters.
            planes: The output filters.
            num_blocks: The number of blocks.
            stride: The stride.
            index: The block index.

        Returns: A ResNet layer.

        """
        layers = nn.ModuleList()
        block_inshape = self.inshapes[index - 1]
        for i in range(num_blocks - 1):  # Create shape preserving blocks.
            newblock = self.block(self.opts, inplanes, inplanes, 1, block_inshape, index=i)
            layers.append(newblock)
        #   block_inshape = self.inshapes[index - 1]  # Compute the shape upsample to.
        newblock = self.block(self.opts, inplanes, planes, stride, block_inshape,
                              index=num_blocks - 1)  # Create upsampling block.
        layers.append(newblock)
        return layers

    def forward(self, inputs: tuple[torch, torch, torch]) -> list[torch, list[torch]]:
        """
        Args:
            inputs: The output from the BU1 stream, flag the task+arg flag , laterals_in, the laterals from the
            BU1 stream.

        Returns: The TD output, lateral connections for BU2.

        """
        bu_out, flag, laterals_in = inputs
        laterals_out = []
        if self.use_initial_emb:
            x = self.InitialTaskEmbedding(bu_out, flag)  # Compute the initial task embedding.
        else:
            x = bu_out
        laterals_out.append([x])
        x = self.top_upsample(x)  # Upsample to the shape before the avgpool in the BU1 stream.
        use_lateral = laterals_in is not None and self.use_lateral
        # If we have lateral connections and use them we invert them to match our size else None
        reverse_laterals_in = laterals_in[::-1] if use_lateral else None
        for layer_id, layer in enumerate(self.alllayers):  # Iterating over all layers in the stream.
            layer_lats_out = []  # The lateral connections for the BU2 stream.
            for block_id, block in enumerate(layer):  # Iterating over all blocks in the layer.
                block_id_rev = len(layer) - block_id - 1
                reverse_cur_lat_in = get_laterals(reverse_laterals_in, layer_id + 1, block_id_rev)[
                                     ::-1]  # Inverting the laterals to match the desired shape.
                # Compute the block output using x, and the lateral connections.
                x, block_lats_out = block(x, reverse_cur_lat_in)
                layer_lats_out.append(block_lats_out)  # Add the lateral output for the next stream.
            reverse_layer_lats_out = layer_lats_out[::-1]  # Reverse to match the shape for the next stream.
            laterals_out.append(reverse_layer_lats_out)  # Add all the layer's laterals.
        lateral_in = reverse_laterals_in[-1][0] if use_lateral else None
        if lateral_in is not None:
            x = self.bot_lat(x, lateral_in)  # Apply last lateral connection.
        laterals_out.append([x])
        outs = [x, laterals_out[::-1]]  # Return the stream output and the lateral connections.
        return outs


class BUTDModel(nn.Module):
    def __init__(self, opts: argparse):
        """
        The main_fashion model.
        The full BU-TD model with possible embedding for continual learning.
        If shared is True, BU1 and BU2 have the same conv layers, otherwise independent layers.
        Args:
            opts: Model options.
        """
        super(BUTDModel, self).__init__()
        self.opts = opts  # Save the model opts.
        self.model_flag = opts.model_flag  # The model type
        self.use_bu1_loss = opts.use_bu1_loss  # Whether to use the Occurrence loss.
        self.use_lateral_butd = opts.use_lateral_butd  # Whether to use the BU1 -> TD laterals.
        self.use_lateral_tdbu = opts.use_lateral_tdbu  # Whether to use the TD -> BU2 laterals.
        # If we use, we create occurrence head.
        if self.use_bu1_loss:
            self.occhead = OccurrenceHead(opts)  # TODO - CHANGE TO OCCURRENCE_HEAD
        bu_shared = BUStreamShared(opts)  # The shared part between BU1, BU2.
        shapes = bu_shared.inshapes  # The model output layers shape.
        #   self.bu_inshapes = bu_shared.inshapes
        self.bumodel1 = BUStream(opts, bu_shared, is_bu2=False)  # The BU1 stream, with no task embedding.
        self.tdmodel = TDModel(opts, shapes)  # The TD stream.
        # If shared is true, the shared part is the same, otherwise we create another shared part.
        if opts.shared:
            bu_shared2 = bu_shared  # Use existing.
        else:
            bu_shared2 = BUStreamShared(opts)  # Create new shared part.
        self.bumodel2 = BUStream(opts, bu_shared2, is_bu2=True)  # The BU2 stream.
        # To save the taskhead parameters.
        self.transfer_learning = [[[] for _ in range(opts.ndirections)] for _ in range(opts.ntasks)]
        self.Head = MultiTaskHead(opts,
                                  self.transfer_learning)  # The task-head to transform to the number of classes.
        if self.model_flag is Flag.CL:  # Storing the Task embedding.
            self.TE = self.bumodel2.task_embedding
            # Store the argument embedding.
            if opts.ds_type is DsType.Omniglot:
                self.argument_embedding = self.tdmodel.argument_embedding
        self.trained_tasks = list()

    def forward(self, samples: inputs_to_struct) -> list[torch]:
        """
        Args:
            samples: The model input.

        Returns: The output from all streams.

        """
        images = samples.image
        flags = samples.flag
        # The input to the BU1 stream is just the images and the flags.
        bu1_model_inputs = [images, flags, None]
        bu_out, bu_laterals_out = self.bumodel1(bu1_model_inputs)
        # If true we add it to the outputs, and apply loss on it.
        if self.use_bu1_loss:
            occurrence_out = self.occhead(bu_out)  # Compute the occurrence head output.
        else:
            occurrence_out = None
        td_model_inputs = [bu_out, flags]
        # Add the BU1 lateral connections to the TD inputs,
        if self.use_lateral_butd:
            td_model_inputs += [bu_laterals_out]
        else:
            td_model_inputs += [None]
        # The input to the TD stream is the bu_out, flags, the lateral connections.
        td_outs = self.tdmodel(td_model_inputs)
        _, td_laterals_out = td_outs
        bu2_model_inputs = [images, flags]
        if self.use_lateral_tdbu:
            bu2_model_inputs += [td_laterals_out]
        else:
            bu2_model_inputs += [None]
        # The input to the BU2 stream is the images, flags, the lateral connections.
        bu2_out, bu2_laterals_out = self.bumodel2(bu2_model_inputs)
        head_input = [bu2_out, flags]
        # The input to the head is the flags, and BU2 output.
        task_out = self.Head(head_input)  # Compute the classification layer.
        outs = [occurrence_out, bu_out, bu2_out, task_out]
        return outs  # Return all the outputs from all streams.

    def forward_and_out_to_struct(self, inputs: inputs_to_struct):
        """
        Do forward pass, and make the output a struct.
        Args:
            inputs: The model input.

        Returns: The model output in the desired head.

        """
        outs = self.forward(inputs)  # Forward.
        return self.opts.outs_to_struct(outs)  # Make the output, a struct.

    def update_task_set(self, task):
        self.trained_tasks.append(task)


class ResNet(nn.Module):
    """
    A ResNet model.
    """

    def __init__(self, opts: argparse):
        """
        Pure ResNet model, used usually as a baseline to the BU-TD model.
        Args:
            opts: The model options.
        """
        super(ResNet, self).__init__()
        self.opts = opts  # The model opts.
        self.ntasks = opts.ntasks  # The number of tasks.
        self.ndirections = opts.ndirections  # The number of directions.
        self.feature_extractor = BUModel(opts,
                                         use_task_embedding=False)  # Create the backbone without the task embedding.
        self.TL = [[[] for _ in range(opts.ndirections)] for _ in range(opts.ntasks)]  # Store the read-out parameters.
        self.classifier = MultiTaskHead(opts, self.TL)  # The classifier head.
        self.trained_tasks: list[int, tuple[int, int]] = list()

    def forward(self, samples: inputs_to_struct):
        """
        Compute model output, including model features and classes
        Args:
            samples: The model inputs.

        Returns: Compute the features and the classification head.

        """
        flags = samples.flag  # The flag.
        bu_out, _ = self.feature_extractor([samples.image, samples.flag, None])  # Compute the features.
        task_out = self.classifier((bu_out, flags))  # The classifier.
        return [None, None, bu_out, task_out]

    def forward_and_out_to_struct(self, samples: inputs_to_struct):
        """
        Args:
            samples: The model inputs.

        Returns: The output struct.

        """
        outs = self.forward(samples)  # The model output.
        return self.opts.outs_to_struct(outs)  # Making the struct.

    def update_task_set(self, task):
        self.trained_tasks.append(task)

    def get_specific_head(self, task_id, direction_tuple):
        learned_params = []
        learned_params.extend(self.feature_extractor.parameters())
        direction_id, _ = tuple_direction_to_index(self.opts.num_x_axis, self.opts.num_y_axis,
                                                   direction=direction_tuple, ndirections=self.opts.ndirections,
                                                   task_id=task_id)
        learned_params.extend(self.TL[task_id][direction_id])
        return learned_params
