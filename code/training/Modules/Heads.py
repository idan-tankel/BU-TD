"""
Here we define the heads, including single head, multi head and occurrence head.
"""
import argparse
from typing import Union, Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..Data.Data_params import Flag
from ..Utils import tuple_to_direction
from ..Data.Structs import inputs_to_struct


# Here we define the task-head modules.

class HeadMultiClass(nn.Module):
    """
    Multi Class Head.
    Support single output or output per class.
    """

    def __init__(self, infilters: int, nclasses: int, num_heads: int):
        """
        Multi Class Head.
        Args:
            infilters: The infilters.
            nclasses: The number of classes.
            num_heads: The number of heads.
        """
        super(HeadMultiClass, self).__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(in_features=infilters, out_features=nclasses + 1) for _ in range(num_heads)])

    def forward(self, x: Tensor) -> Tensor:
        """
       Forward the head.
       Args:
           x: The input.

       Returns: The output.

       """
        outs = [layer(x) for layer in self.layers]
        return torch.stack(tensors=outs, dim=-1)


class HeadSingleTask(nn.Module):
    """
    Task head for single task.
    Allocate tasks according to the desired output size.
    if model flag is NOFLAG we allocate subhead for each character.
    """

    def __init__(self, opts: argparse, nclasses: int):
        """
        Args:
            opts: The model options.
            nclasses: The number of classes.
            num_heads: Number of heads for the task.
        """
        super(HeadSingleTask, self).__init__()
        self.opts = opts
        # If The model flag is NOFLAG we allocate for each character a head o.w. according
        # to the nclasses.
        num_head = nclasses if opts.model_flag is Flag.NOFLAG \
            else 1

        infilters = opts.nfilters[-1]  # The input size from the end of the BU2 stream.
        self.layers = nn.ModuleList([HeadMultiClass(infilters, nclasses, num_head)])

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Args:
            inputs: The output from the model.

        Returns: A tensor with the shape of the number of classes.

        """
        x = inputs.squeeze()  # Squeeze the input.
        outs = [layer(x) for layer in self.layers]  # Compute all task-head outputs for all layers.
        return torch.stack(tensors=outs, dim=-1).squeeze() # stacks all tensor into one tensor


class MultiTaskHead(nn.Module):
    """
    # Create task-head per task, task.
    """

    def __init__(self, opts: argparse, transfer_learning_params: Optional[list] = None):
        """
        Multi head task-head allocating for each task and task a single
        task head. Args: opts: The model options. transfer_learning_params: list containing the
        associate taskhead params of the task, task.
        """
        super(MultiTaskHead, self).__init__()
        self.opts = opts  # The options.
        self.ntasks = opts.data_obj.ntasks  # The number of tasks.
        self.ndirections = opts.data_obj.ndirections  # The number of directions.
        self.num_classes = opts.data_obj.nclasses  # The number of classes for each task to create the head
        # according to.
        self.ds_type = self.opts.ds_type  # The data-set type.
        self.taskhead = nn.ModuleList()
        task_head = nn.ModuleList()
        # For each task, task create its task-head according to num_classes.

        for i in range(self.ntasks):
            for j in range(self.ndirections):
                layer = HeadSingleTask(opts=opts, nclasses=self.num_classes[i])  # create a taskhead.
                task_head.append(layer)
                if transfer_learning_params is not None:
                    transfer_learning_params[i][j].extend(layer.parameters())  # Storing the taskhead params.
            self.taskhead.append(task_head)
            task_head = nn.ModuleList()

    def forward(self, inputs: Tuple[Tensor, inputs_to_struct]) -> Tensor:
        """
        Args:
            inputs: The output from BU2, and the flag.

        Returns: A tensor in the desired shape.

        """
        (bu2_out, samples) = inputs
        # In train mode we train only one head.
        direction_id = samples.direction_idx[0]
        task_id = samples.language_index[0]
        task_out = self.taskhead[task_id][direction_id](bu2_out)  # apply the appropriate task-head.

        return task_out


class OccurrenceHead(nn.Module):
    """
    Occurrence head, transforming the BU1 feature to binary classification over all classes.
    """

    def __init__(self, opts: argparse):
        """
        Occurrence head predicting for each character whether it exists in the sample.
        Args:
            opts: The model options.
        """
        super(OccurrenceHead, self).__init__()
        filters = opts.data_obj.nclasses[0]  # The number of binary classifiers needed to recognize all characters.
        infilters = opts.nfilters[-1]  # Output shape from the end of the BU1 stream.
        self.occurrence_transform = nn.Linear(in_features=infilters, out_features=filters)  # The linear transformation.

    def forward(self, bu_out: Tensor) -> Tensor:
        """
        Args:
            bu_out: The BU1 output.

        Returns: The binary classification input.

        """
        x = self.occurrence_transform(input=bu_out)  # Apply the transforms for the BU1 loss.
        return x
