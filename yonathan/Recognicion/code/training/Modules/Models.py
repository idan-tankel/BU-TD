import argparse
import itertools
from typing import Iterator, Union

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
        # The first shape after Conv1.
        inshape = np.array([self.inplanes, np.int(np.ceil(inshape[1] / stride)),
                            np.int(np.ceil(inshape[2] / stride))])
        inshapes = [inshape]
        for k in range(len(opts.strides)):  # For each stride create layer of Blocks.
            nblocks = opts.ns[k]  # The number of blocks in the layer.
            stride = opts.strides[k]  # The stride.
            filters = opts.nfilters[k]  # The number of filters transform to.
            if nblocks > 0:
                self.alllayers.append(self._make_layer(filters, nblocks, stride=stride))  # Making a layer.
                inshape = np.array([filters, np.int(np.ceil(inshape[1] / stride)),
                                    np.int(np.ceil(inshape[2] / stride))])  # Compute the output shape of the layer.
                inshapes.append(inshape)  # Add the shape to the shapes list.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Average pool before the classification.
        if self.use_lateral:
            self.nfilters_top_lat = opts.nfilters[-1]  # The number of filters for the lateral connection.
        self.inshapes = inshapes

    def _make_layer(self, planes: int, nblocks: int, stride: int) -> list[nn.Module]:
        """
        Args:
            planes: Channels_out of the layer.
            nblocks: num blocks in the layer.
            stride: The stride.

        Returns: A ResNet layer.
        """
        layers = [self.block(self.opts, self.inplanes, planes, stride)]  # All Blocks list.
        self.inplanes = planes * self.block.expansion
        for idx in range(0, nblocks - 1):  # Add nblocks - 1 preserving blocks.
            layers.append(self.block(self.opts, self.inplanes, planes, 1))
        return layers


class BUModel(nn.Module):
    # Equivalent to Pure ResNet.
    def __init__(self, opts: argparse, use_task_embedding: bool = False):
        """
        Args:
            opts: The model opts.
            use_task_embedding: Whether to use the task embedding.
        """
        super(BUModel, self).__init__()
        bu_shared = BUStreamShared(opts=opts)
        self.model = BUStream(opts=opts, shared=bu_shared, is_bu2=use_task_embedding)

    def forward(self, inputs: list[torch]) -> list[torch]:
        """
        Args:
            inputs: The input list.

        Returns: The model output.

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
        self.is_bu2 = is_bu2
        self.InitialBlock = BUInitialBlock(opts, shared, is_bu2=is_bu2)
        self.avgpool = shared.avgpool  # Avg pool layer.
        layers = nn.ModuleList()
        for layer_idx, shared_layer in enumerate(
                shared.alllayers):  # For each shared layer we create associate BU layer.
            layers.append(self._make_layer(shared_layer, layer_idx))
        self.alllayers = layers
        if shared.use_lateral and is_bu2:
            self.top_lat = Modulation_and_Lat(opts, shared.nfilters_top_lat)
        init_module_weights(self.modules(), self.model_flag)  # Init the weights.

    def _make_layer(self, blocks: Iterator[nn.Module], layer_id: int) -> nn.ModuleList:
        """
        Args:
            blocks: Shared layers between BU1, BU2.
            layer_id: The layer id for the input shape computing.

        Returns: A ResNet shared layer.

        """
        layers = nn.ModuleList()
        for shared_block in blocks:
            # Create Basic BU blocks for each shared block.
            block_inshape = self.inshapes[layer_id + 1]
            layer = self.block(self.opts, shared_block, block_inshape, self.is_bu2, self.task_embedding)
            layers.append(layer)
        return layers

    def forward(self, inputs: list[torch]) -> tuple:
        """
        The forward includes processing the input and inserting the lateral connections.
        Args:
            inputs: The input, the flags, the lateral connection from TD network.

        Returns: The model output + The lateral connections.

        """
        lateral_layer_id = 0
        # The input is the image, the flag and the lateral from the previous stream(if exist).
        x, flags, laterals_in = inputs
        laterals_out = []  # The laterals for the second stream.
        x = self.InitialBlock((x, flags, laterals_in))  # Compute the initial block in ResNet.
        laterals_out.append([x])
        for layer_id, layer in enumerate(self.alllayers):
            layer_lats_out = []  # The lateral connections for the next stream.
            for block_id, block in enumerate(layer):
                lateral_layer_id = layer_id + 1  # Get layer id.
                cur_lat_in = get_laterals(laterals_in, lateral_layer_id,
                                          block_id)  # Get the laterals associate with the layer, block_id(if exist).
                x, block_lats_out = block((x, flags, cur_lat_in))  # Compute the block with the lateral connection.
                layer_lats_out.append(block_lats_out)
            laterals_out.append(layer_lats_out)
        x = self.avgpool(x)  # Avg pool.
        lateral_in = get_laterals(laterals_in, lateral_layer_id + 1, 0)  # The last lateral connection.
        if self.use_lateral and lateral_in is not None:
            x = self.top_lat((x, lateral_in))  # last lateral connection before the end.
        laterals_out.append([x])
        x = x.squeeze()
        return x, laterals_out


class TDModel(nn.Module):
    def __init__(self, opts: argparse, bu_inshapes: list, avg_pool_size: tuple):
        """
        Args:
            opts: The model opts.
            bu_inshapes: The BU shapes.
            avg_pool_size: The avg_pool_size.
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
        self.task_embedding = [[] for _ in range(self.ndirections)]
        self.argument_embedding = [[] for _ in range(self.ntasks)]
        self.use_initial_emb = self.model_flag is not Flag.NOFLAG
        if self.use_initial_emb:
            self.InitialTaskEmbedding = InitialEmbeddingBlock(opts)  # Initial TD block.
            # Upsample layer to make at of the shape before the avgpool.
        self.top_upsample = nn.Upsample(scale_factor=avg_pool_size, mode='bilinear',
                                        align_corners=False)
        self.alllayers = nn.ModuleList()
        # Create layers.
        for k in range(len(opts.strides) - 1, 0, -1):
            nblocks = opts.ns[k]
            stride = opts.strides[k]
            filters = opts.nfilters[k - 1]
            if nblocks > 0:
                self.alllayers.append(self._make_layer(filters, nblocks, stride=stride,
                                                       index=k))  # Create the exact opposite layers of the BU1 stream.
        if self.use_lateral:
            self.bot_lat = Modulation_and_Lat(opts, opts.nfilters[0])
        # Copy the argument embedding.
        if opts.ds_type is DsType.Omniglot and self.model_flag is Flag.CL:
            for j in range(self.ntasks):
                self.argument_embedding[j].extend(self.InitialTaskEmbedding.top_td_arg_emb[j].parameters())

        # copy the task embedding.
     #   if self.model_flag is Flag.CL:
#            self.task_embedding = [[self.InitialTaskEmbedding.top_td_task_emb[i]] for i in
       #                            range(self.ndirections)]
        init_module_weights(self.modules(), self.model_flag)  # Init the weights.

    def _make_layer(self, planes: int, num_blocks: int, stride: int = 1, index: int = 0):
        """
        Args:
            planes: The outplanes.
            num_blocks: The number of blocks.
            stride: The stride.
            index: The block index.

        Returns: A ResNet layer.

        """
        layers = nn.ModuleList()
        for _ in range(1, num_blocks):  # Create shape preserving blocks.
            block_inshape = self.inshapes[index - 1]
            newblock = self.block(self.opts, self.inplanes, self.inplanes, 1, block_inshape)
            layers.append(newblock)

        block_inshape = self.inshapes[index - 1]  # Compute the shape upsample to.
        newblock = self.block(self.opts, self.inplanes, planes, stride, block_inshape)  # Create upsample block.
        layers.append(newblock)
        self.inplanes = planes * self.block.expansion
        return layers

    def forward(self, inputs: tuple[torch]) -> list[torch, list[torch]]:
        """
        Args:
            inputs: The output from the BU1 stream, flag the task+arg flag , laterals_in, the laterals from the
            BU1 stream.

        Returns: The td outputs + lateral connections foR bu1.

        """
        bu_out, flag, laterals_in = inputs
        laterals_out = []
        if self.use_initial_emb:
            x = self.InitialTaskEmbedding((bu_out, flag))  # Compute the initial task embedding.
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
                reverse_cur_lat_in = get_laterals(reverse_laterals_in, layer_id + 1, block_id)[
                                     ::-1]  # Inverting the laterals to match the desired shape.
                # Compute the block output using x, the flag and the lateral connections.
                x, block_lats_out = block((x, flag, reverse_cur_lat_in))
                layer_lats_out.append(block_lats_out)  # Add the lateral output for the next stream.
            reverse_layer_lats_out = layer_lats_out[::-1]
            laterals_out.append(reverse_layer_lats_out)  # Add all the layer's laterals.
        lateral_in = reverse_laterals_in[-1][0] if use_lateral else None
        if lateral_in is not None:
            x = self.bot_lat((x, lateral_in))  # Compute lateral connection + channel modulation.
        laterals_out.append([x])
        outs = [x, laterals_out[::-1]]  # Output the output of the stream + the lateral connections.
        return outs


class BUTDModel(nn.Module):
    def __init__(self, opts: argparse):
        """
        The main model.
        The full BU-TD model with possible embedding for continual learning.
        Args:
            opts: Model options.
        """
        super(BUTDModel, self).__init__()
        self.task_embedding = None
        self.argument_embedding = None
        self.opts = opts
        self.use_lateral_butd = opts.use_lateral_butd
        self.use_lateral_tdbu = opts.use_lateral_tdbu
        self.model_flag = opts.model_flag  # The model type
        self.use_bu1_loss = opts.use_bu1_loss  # Whether to use the Occurrence loss.
        self.ds_type = opts.ds_type
        if self.use_bu1_loss:
            self.occhead = OccurrenceHead(opts)
        bu_shared = BUStreamShared(opts)  # The shared part between BU1, BU2.
        shapes = bu_shared.inshapes
        avg_pool_size = tuple(shapes[-1][1:])
        self.bu_inshapes = bu_shared.inshapes
        self.bumodel1 = BUStream(opts, bu_shared, is_bu2=False)  # The BU1 stream.
        self.tdmodel = TDModel(opts, shapes, avg_pool_size)  # The TD stream.
        self.bumodel2 = BUStream(opts, bu_shared, is_bu2=True)  # The BU2 stream.
        self.transfer_learning = [[[] for _ in range(opts.ndirections)] for _ in range(opts.ntasks )]
        self.Head = MultiTaskHead(opts,
                                  self.transfer_learning)  # The task-head to transform to the number of classes.
        if self.model_flag is Flag.CL:  # Storing the Task embedding.
            TE_bumodel = self.bumodel2.task_embedding
            TE_tdmodel = self.tdmodel.task_embedding
            self.TE = [list(itertools.chain.from_iterable(x)) for x in zip(TE_bumodel, TE_tdmodel)]
            if self.ds_type is DsType.Omniglot:
                self.argument_embedding = self.tdmodel.argument_embedding

    def forward(self, samples: inputs_to_struct) -> list[torch]:
        """
        Args:
            samples: The model input.

        Returns: The output from all streams.

        """
        #  samples = self.inputs_to_struct(inputs)  # Making the input to a struct.
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
        bu2_out = bu2_out
        head_input = [bu2_out, flags]
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
        self.opts = opts
        self.ntasks = opts.ntasks
        self.ndirections = opts.ndirections
        self.feature = BUModel(opts, use_task_embedding=False)  # Create the backbone without the task embedding.
        self.TL = [[[] for _ in range(opts.ndirections)] for _ in range(opts.ntasks )]  # Store the read-out parameters.
        self.classifier = MultiTaskHead(opts, self.TL)  # The classifier head.
        self.trained_tasks:set[int,tuple[int,int]] = set()

    def forward_features(self, samples: inputs_to_struct, features: torch, head: Union[None, int] = None) -> torch:
        """
        Needed for switching a task-head given computed features.
        Args:
            samples: The model input.
            features: The feature output.
            head: The head index if we want to switch.

        Returns: The classification.

        """
        flags = samples.flag
        task_out = self.taskhead((features, flags), head)
        return task_out

    def compute_features(self, samples: inputs_to_struct) -> torch:
        """
        Args:
            samples: Given samples, we compute only the features(the layer before the classes).

        Returns: The computed features.

        """
        images = samples.image  # The images.
        flags = samples.flag  # The flag.
        model_inputs = [images, flags, None]  # Make the model input.
        features, _ = self.feature(model_inputs)  # The features.
        return features

    def forward(self, samples: inputs_to_struct):
        """
        Args:
            samples: The model inputs.

        Returns: Compute the features and the classification head.

        """
        flags = samples.flag  # The flag.
        bu_out = self.compute_features(samples)  # Compute the features.
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

    def Get_learned_params(self, task_id: int, direction_id: tuple) -> list:
        """
        For a task_id and direction_id return the backbone parameters,
        and the specific task head(to avoid possible changes of other task heads).
        Args:
            task_id: The task id.
            direction_id: The direction id.

        Returns: The learnable parameters.

        """
        learned_param = []
        learned_param.extend(self.feature.parameters())
        head_idx = tuple_direction_to_index(self.opts.num_x_axis, self.opts.num_y_axis, direction_id, self.ndirections,
                                            task_id)
        learned_param.extend(self.classifier[head_idx].parameters())
        return learned_param

    def update_task_set(self,task):
        self.trained_tasks.add(task)
