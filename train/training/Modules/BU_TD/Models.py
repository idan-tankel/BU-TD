"""
Here we define the models.
It includes BUTDModel, BUModel and Pure ResNet.
"""
import argparse
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ...Data.Enums import Flag, DsType
from ...Data.Structs import Spatial_Relations_inputs_to_struct, outs_to_struct
from .Heads import MultiTaskHead, OccurrenceHead
from .Blocks import BUInitialBlock, InitialEmbeddingBlock, Modulation_and_Lat, \
    BasicBlockBUShared, BasicBlockBU, BasicBlockTD
from ...Utils import get_laterals, tuple_direction_to_index
from ..Module_Blocks import layer_with_modulation_and_masking


def init_module_weights(modules: Iterator[nn.Module]) -> None:
    """
    Initializing the module weights according to the original BU_TD paper.
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
        elif isinstance(m, Modulation_and_Lat):
            nn.init.xavier_uniform_(m.side)


        elif isinstance(m, layer_with_modulation_and_masking):
            for param in m.modulation.modulation:
                nn.init.kaiming_normal_(param, nonlinearity="relu")


class BUStreamShared(nn.Module):
    """
    Here we define the shared layers between BU1, BU2.
    """

    def __init__(self, opts: argparse):
        """
        The Shared stream between BU1, BU2, contain only conv layers.
        Args:
            opts: Model options.
        """

        super(BUStreamShared, self).__init__()
        self.opts = opts  # The model options.
        self.block = BasicBlockBUShared  # The block type.
        Model_inshape = np.array(opts.data_obj['image_size'])  # The image resolution.
        filters = opts.data_obj['nfilters']
        initial_ks = opts.data_obj['initial_ks']
        strides = opts.data_obj['strides']
        initial_pad = opts.data_obj['initial_pad']
        num_blocks = opts.data_obj['num_blocks']
        self.conv1 = nn.Conv2d(in_channels=Model_inshape[0], out_channels=filters[0],
                               kernel_size=initial_ks, stride=strides[0],
                               padding=initial_pad)  #
        # The first conv layer as in ResNet.
        # The first shape after Conv1.
        inshape = np.array([filters[0], np.int(np.ceil(Model_inshape[1] / strides[0])),
                            np.int(np.ceil(Model_inshape[2] / strides[0]))])
        self.inshapes = [inshape]  # list that should contain all layer output shapes.
        self.alllayers = []  # All layers list.
        inplanes = filters[0]
        for k in range(len(num_blocks)):  # For each stride create layer of Blocks.
            nblocks = num_blocks[k]  # The number of blocks in the layer.
            stride = strides[k + 1]  # The stride.
            filter = filters[k + 1]  # The number of filters transforms to.
            self.alllayers.append(
                self._make_layer(inplanes=inplanes, planes=filter, nblocks=nblocks, stride=stride))  #
            # Making a layer.
            # Compute the output shape of the layer.
            inshape = np.array([filter, np.int(np.ceil(inshape[1] / stride)),
                                np.int(np.ceil(inshape[2] / stride))])
            self.inshapes.append(inshape)  # Add the shape to the shapes list.
            inplanes = filter  # Update the future inplanes to be the current filters.

    def _make_layer(self, inplanes: int, planes: int, nblocks: int, stride: int) \
            -> list[nn.Module]:
        """
        Args:
            inplanes: Channels_in of the layer.
            planes: Channels_out of the layer.
            nblocks: num blocks in the layer.
            stride: The stride.

        Returns: A ResNet layer.
        """
        planes = planes * self.block.expansion
        layers = [
            self.block(opts=self.opts, in_channels=inplanes, out_channels=planes, stride=stride)]
        # Create a
        # block.
        for idx in range(0, nblocks - 1):  # Add nblocks - 1 preserving blocks.
            layers.append(self.block(opts=self.opts, in_channels=planes, out_channels=planes, stride=1))
        return layers


# TODO - DELETE THIS.
class BUModel(nn.Module):
    """
    Equivalent to Pure ResNet.
    """

    def __init__(self, opts: argparse, use_task_embedding: bool = False):
        """
        Args:
            opts: The model opts.
            use_task_embedding: Whether to create the task embedding.
        """
        super(BUModel, self).__init__()
        bu_shared = BUStreamShared(opts=opts)  # The shared model part.
        self.model = BUStream(opts=opts, shared=bu_shared, is_bu2=use_task_embedding)  # Create the model.

    def forward(self, inputs: list[Tensor]) -> list[Tensor]:
        """
        Args:
            inputs: The model input.

        Returns: The model output.

        """
        return self.model(inputs=inputs)


class BUStream(nn.Module):
    """
    The BU stream model.
    """

    def __init__(self, opts: argparse, shared: nn.Module, is_bu2: bool = False):
        """
        Args:
            opts: The model options.
            shared: The shared part between BU1, BU2.
            is_bu2: Whether the stream is BU1 or BU2.
        """
        super(BUStream, self).__init__()
        self.opts = opts  # Save the model opts.
        self.block = BasicBlockBU  # The basic block type.
        self.task_embedding = [[] for _ in range(opts.data_obj['ndirections'])]  # List should contain the task
        self.masks = [[] for _ in range(opts.data_obj['ndirections'])]  # List should contain the task
        # embedding.
        self.inshapes = shared.inshapes  # The output shape of all layers.
        self.use_lateral = opts.data_obj['use_lateral_tdbu']  # Whether to use the TD -> BU2 laterals.
        self.is_bu2 = is_bu2  # Save whether we are on the BU2 stream.
        # The initial block, getting the image as an input.
        self.InitialBlock = BUInitialBlock(opts=opts, shared=shared,
                                           is_bu2=is_bu2)
        self.alllayers = nn.ModuleList()
        # For each shared layer we create associate BU layer.
        for layer_idx, shared_layer in enumerate(shared.alllayers):
            self.alllayers.append(self._make_layer(blocks=shared_layer, layer_id=layer_idx))
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # Avg pool layer.
        # Last lateral connection fot the beginning of the TD-stream.
        if self.use_lateral and is_bu2:
            self.top_lat = Modulation_and_Lat(opts=opts, nfilters=opts.data_obj['nfilters'][-1])
        # init_module_weights(self.modules())  # Initialize the weights.

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
            block_inshape = self.inshapes[
                layer_id + 1]  # The block input shape, needed for the task embedding.
            layer = self.block(opts=self.opts, shared=shared_block, block_inshapes=block_inshape, is_bu2=self.is_bu2,
                               task_embedding=self.task_embedding, masks=self.masks)
            layers.append(layer)  # Add the layer.
        return layers

    def forward(self, inputs: list[Spatial_Relations_inputs_to_struct, Tensor]) -> tuple[Tensor, list[list[Tensor]]]:
        """
        The forward includes processing the input and inserting the lateral connections.
        Args:
            inputs: The input, the samples, the lateral connection from TD network.

        Returns: The model output + The lateral connections.

        """
        # The input is the image, the flag and the lateral connections from the previous stream(if exist).
        samples, laterals_in = inputs
        x = samples.image
        laterals_out: list[list[Tensor]] = []  # The laterals for the second stream.
        x: Tensor = self.InitialBlock(x, samples, laterals_in)  # Compute the initial block in ResNet.
        laterals_out.append([x])
        for layer_id, layer in enumerate(self.alllayers):
            layer_lats_out: list[Tensor] = []  # The lateral connections for the next stream.
            for block_id, block in enumerate(layer):
                lateral_layer_id = layer_id + 1  # Get layer id.
                # Get the laterals associate with the layer, block_id(if exist).
                cur_lat_in = get_laterals(laterals=laterals_in, layer_id=lateral_layer_id,
                                          block_id=block_id)
                x, block_lats_out = block(x=x, samples=samples,
                                          laterals_in=cur_lat_in)  # Compute the block with the lateral connection.
                layer_lats_out.append(block_lats_out)
            laterals_out.append(layer_lats_out)
        x = self.avgpool(input=x)  # Avg pool.
        lateral_in: Tensor = get_laterals(laterals=laterals_in,
                                          layer_id=len(self.alllayers) + 1)  # The last lateral connection.
        if self.use_lateral and lateral_in is not None:
            x: Tensor = self.top_lat(x=x, samples=samples, lateral=lateral_in)  # last lateral connection before the
            # end.
        laterals_out.append([x])  # Add to laterals.
        x = x.squeeze()  # Squeeze to be a vector.
        return x, laterals_out


class TDModel(nn.Module):
    """
    TD model.
    Getting the argument and lateral connections as an input.
    """

    def __init__(self, opts: argparse, bu_inshapes: list):
        """
        Args:
            opts: The model opts.
            bu_inshapes: The BU shapes.
        """
        super(TDModel, self).__init__()
        self.opts = opts  # Save the opts.
        self.block = BasicBlockTD  # The block type
        self.use_lateral = opts.data_obj['use_lateral_butd']  # Whether to use the BU1 -> TD laterals.
        self.ntasks = opts.data_obj['ntasks']  # The number of tasks
        self.ndirections = opts.data_obj['ndirections']
        self.inshapes = bu_inshapes  # The model layers output.
        self.argument_embedding = [[] for _ in range(self.ntasks)]
        num_blocks = opts.data_obj['num_blocks']
        strides = opts.data_obj['strides']
        nfilters = opts.data_obj['nfilters']
        self.use_embedding = opts.use_embedding
        self.task_embedding = [[] for _ in range(self.ndirections)]
        self.masks = [[] for _ in range(self.ndirections)]
        # Only in Omniglot we have several tasks, with different argument embeddings.
        self.use_initial_emb = opts.model_flag is not Flag.NOFLAG
        if self.use_initial_emb:
            # Create the initial TD block.
            self.InitialTaskEmbedding = InitialEmbeddingBlock(opts=opts)
        # Upsample layer to match the shape before the avgpool.
        shape_before_avg_pool = tuple(
            bu_inshapes[-1][1:])  # The last shape of BU1, we task the inner shape.
        inshape = bu_inshapes[-1][0]
        self.top_upsample = nn.Upsample(scale_factor=shape_before_avg_pool, mode='bilinear',
                                        align_corners=False)
        # List contain all layers.
        self.alllayers = nn.ModuleList()
        # Create layers.
        for k in range(len(num_blocks) - 1, -1, -1):
            nblocks = num_blocks[k]  # The number of blocks in the layer.
            stride = strides[k + 1]  # The current stride.
            filters = nfilters[k]  # The number of filters.
            self.alllayers.append(self._make_layer(inplanes=inshape, planes=filters, num_blocks=nblocks, stride=stride,
                                                   index=k))
            # Create the exact opposite layers of the BU1 stream.
            inshape = filters  # The current output filters, is the next layer input filter.
        # The last lateral connection
        if self.use_lateral:
            self.bot_lat = Modulation_and_Lat(opts=opts, nfilters=nfilters[0])
        # Copy the argument embedding.
        if opts.model_flag is Flag.CL:
            for j in range(self.ntasks):
                self.argument_embedding[j].extend(self.InitialTaskEmbedding.top_td_arg_emb[j].parameters())

    def _make_layer(self, inplanes: int, planes: int, num_blocks: int, stride: int = 1,
                    index: int = 0):
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
        block_inshape = self.inshapes[index]
        for i in range(num_blocks - 1):  # Create shape preserving blocks.
            newblock = self.block(opts=self.opts, in_channels=inplanes, out_channels=inplanes, stride=1,
                                  block_inshape=block_inshape, index=i)
            layers.append(newblock)
        newblock = self.block(opts=self.opts, in_channels=inplanes, out_channels=planes, stride=stride,
                              block_inshape=block_inshape,
                              index=num_blocks - 1,task_embedding=self.task_embedding,
                              masks=self.masks)  # Create upsampling block.
        layers.append(newblock)
        return layers

    def forward(self, inputs: tuple[Tensor, Spatial_Relations_inputs_to_struct, Tensor]) -> list[
        Tensor, list[list[Tensor]]]:
        """
        Args: inputs: The output from the BU1 stream, flag the task+arg flag , laterals_in, the laterals
        from the BU1 stream.

        Returns: The TD output, lateral connections for BU2.

        """
        bu_out, samples, laterals_in = inputs
        # flags = samples.char_flags
        laterals_out = []
        if self.use_initial_emb and self.use_embedding:
            x = self.InitialTaskEmbedding(bu_out=bu_out, samples=samples)  # Compute the initial task
            # embedding.
        else:
            x = bu_out.unsqueeze(2).unsqueeze(2)
        laterals_out.append([x])
        x = self.top_upsample(x)  # Upsample to the shape before the avgpool in the BU1 stream.
        use_lateral = laterals_in is not None and self.use_lateral
        # If we have lateral connections and use them we invert them to match our size else None
        reverse_laterals_in = laterals_in[::-1] if use_lateral else None
        for layer_id, layer in enumerate(self.alllayers):  # Iterating over all layers in the stream.
            layer_lats_out = []  # The lateral connections for the BU2 stream.
            for block_id, block in enumerate(layer):  # Iterating over all blocks in the layer.
                block_id_rev = len(layer) - block_id - 1
                reverse_cur_lat_in = get_laterals(laterals=reverse_laterals_in, layer_id=layer_id + 1,
                                                  block_id=block_id_rev)[
                                     ::-1]  # Inverting the laterals to match the desired shape.
                # Compute the block output using x, and the lateral connections.
                x, block_lats_out = block(x=x, samples=samples, laterals=reverse_cur_lat_in)
                layer_lats_out.append(block_lats_out)  # Add the lateral output for the next stream.
            reverse_layer_lats_out = layer_lats_out[::-1]  # Reverse to match the shape for the next stream.
            laterals_out.append(reverse_layer_lats_out)  # Add all the layer's laterals.
        lateral_in = reverse_laterals_in[-1][0] if use_lateral else None
        if lateral_in is not None:
            x = self.bot_lat(x=x, samples=samples, lateral=lateral_in)  # Apply last lateral connection.
        laterals_out.append([x])
        outs = [x, laterals_out[::-1]]  # Return the stream output and the lateral connections.
        return outs


class BUTDModel(nn.Module):
    """
    The main model.
    The full BU_TD model with possible embedding for continual learning.
    """

    def __init__(self, opts: argparse):
        """

        If shared is True, BU1 and BU2 have the same conv layers, otherwise independent layers.
        Args:
            opts: Model options.
        """
        super(BUTDModel, self).__init__()
        self.opts = opts  # Save the model opts.
        self.model_flag = opts.model_flag  # The model type
        self.use_bu1_loss = opts.data_obj['use_bu1_loss']  # Whether to use the Occurrence loss.
        self.use_lateral_butd = opts.data_obj['use_lateral_butd']  # Whether to use the BU1 -> TD laterals.
        # Whether to use the TD -> BU2 laterals.
        self.use_lateral_tdbu = opts.data_obj['use_lateral_tdbu'] and opts.model_flag is not Flag.NOFLAG
        # If we use, we create occurrence head.
        if self.use_bu1_loss:
            self.occhead = OccurrenceHead(opts=opts)
        bu_shared = BUStreamShared(opts=opts)  # The shared part between BU1, BU2.
        shapes = bu_shared.inshapes  # The model output layers shape.
        #   self.bu_inshapes = bu_shared.inshapes
        self.bumodel1 = BUStream(opts=opts, shared=bu_shared)  # The BU1 stream, with no task embedding.
        if self.model_flag is not Flag.NOFLAG:
            self.tdmodel = TDModel(opts=opts, bu_inshapes=shapes)  # The TD stream.
        # If shared is true, the shared part is the same, otherwise we create another shared part.
        if opts.data_obj['shared']:
            bu_shared2 = bu_shared  # Use existing.
        else:
            bu_shared2 = BUStreamShared(opts=opts)  # Create new shared part.
        self.bumodel2 = BUStream(opts=opts, shared=bu_shared2, is_bu2=True)  # The BU2 stream.
        # To save the taskhead parameters.
        self.transfer_learning = [[[] for _ in range(opts.data_obj['ndirections'])] for _ in range(
            opts.data_obj['ntasks'])]
        # The task-head to transforms to the number of classes.
        self.Head = MultiTaskHead(opts=opts,
                                  transfer_learning_params=self.transfer_learning)
        self.tdmodel_mod = self.tdmodel.task_embedding
        self.tdmode_arg_emb = self.tdmodel.InitialTaskEmbedding.parameters()
        if self.model_flag is Flag.CL:  # Storing the Task embedding.
            self.TE = self.bumodel2.task_embedding
            self.masks = self.bumodel2.masks
            # Store the argument embedding.
            if opts.ds_type is DsType.Omniglot:
                self.argument_embedding = self.tdmodel.argument_embedding
        self.trained_tasks = list()
        init_module_weights(modules=self.modules())  # Initialize the weights.

    def forward(self, samples: Spatial_Relations_inputs_to_struct) -> list[torch]:
        """
        Args:
            samples: The model input.

        Returns: The output from all streams.

        """
        # The input to the BU1 stream is just the images and the samples.
        bu1_model_inputs = [samples, None]
        bu_out, bu_laterals_out = self.bumodel1(inputs=bu1_model_inputs)
        # If true we add it to the outputs, and apply loss on it.
        if self.use_bu1_loss:
            occurrence_out = self.occhead(bu_out=bu_out)  # Compute the occurrence head output.
        else:
            occurrence_out = None
        td_model_inputs = [bu_out, samples]
        # Add the BU1 lateral connections to the TD inputs,
        if self.use_lateral_butd:
            td_model_inputs += [bu_laterals_out]
        else:
            td_model_inputs += [None]
        # The input to the TD stream is the bu_out, samples, the lateral connections.
        if self.model_flag is not Flag.NOFLAG:
            td_outs = self.tdmodel(inputs=td_model_inputs)
            _, td_laterals_out = td_outs
        else:
            td_laterals_out = None
        bu2_model_inputs = [samples]
        if self.use_lateral_tdbu:
            bu2_model_inputs += [td_laterals_out]
        else:
            bu2_model_inputs += [None]
        # The input to the BU2 stream is the images, samples, the lateral connections.
        bu2_out, bu2_laterals_out = self.bumodel2(inputs=bu2_model_inputs)
        head_input = [bu2_out, samples]
        # The input to the head is the samples, and BU2 output.
        task_out = self.Head(inputs=head_input)  # Compute the classification layer.
        outs = [occurrence_out, bu_out, bu2_out, task_out]
        return outs  # Return all the outputs from all streams.

    def forward_and_out_to_struct(self, inputs: Spatial_Relations_inputs_to_struct) -> outs_to_struct:
        """
        Do forward pass, and make the output a struct.
        Args:
            inputs: The model input.

        Returns: The model output in the desired head.

        """
        outs = self.forward(samples=inputs)  # Forward.
        return self.opts.outs_to_struct(outs=outs)  # Make the output, a struct.

    def update_task_list(self, task: tuple) -> None:
        """
        Update the tasks list.
        Args:
            task: The task.

        """

        self.trained_tasks.append(task)

    def feature_extractor(self):
        """
        Return the non-classifier parameters.
        Returns:

        """
        return nn.ModuleList([self.bumodel1, self.bumodel2, self.tdmodel])

    def get_specific_head(self, task_id: int, direction_tuple: tuple[int, int]) -> list[nn.Parameter]:
        """
        Get the specific head and the feature parameters.
        Args:
            task_id: The task id.
            direction_tuple: The task tuple

        Returns: The learned parameters.

        """
        learned_params = []
        learned_params.extend(self.bumodel1.parameters())
        learned_params.extend(self.bumodel2.parameters())
        learned_params.extend(self.tdmodel.parameters())
        learned_params = list(set(learned_params))
        direction_id, _ = tuple_direction_to_index(num_x_axis=self.opts.num_x_axis, num_y_axis=self.opts.num_y_axis,
                                                   direction=direction_tuple, ndirections=self.opts.ndirections,
                                                   language_idx=task_id)
        learned_params.extend(self.transfer_learning[task_id][direction_id])
        return learned_params
