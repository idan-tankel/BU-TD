import argparse
from typing import Union

import torch
from torch import Tensor
import torch.nn as nn

from training.Data.Data_params import Flag
from training.Utils import Flag_to_task


# Here we define the task-head modules.

class HeadSingleTask(nn.Module):
    """
    Task head for single task.
    Allocate tasks according to the desired output size.
    if model flag is NOFLAG we allocate subhead for each character.
    """

    def __init__(self, opts: argparse, nclasses: int, num_heads=1) -> None:
        """
        Args:
            opts: The model options.
            nclasses: The number of classes.
            num_heads: Number of heads for the task.
        """
        super(HeadSingleTask, self).__init__()
        self.opts = opts
        num_heads = nclasses if opts.model_flag is Flag.NOFLAG \
            else num_heads  # If The model flag is NOFLAG we allocate for each character a head o.w. according to the
        # nclasses.
        infilters = opts.nfilters[-1]  # The input size from the end of the BU2 stream.
        self.layers = nn.ModuleList([nn.Linear(infilters, nclasses + 1) for _ in range(num_heads)])

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Args:
            inputs: The output from the model.

        Returns: A tensor with the shape of the number of classes.

        """
        x = inputs.squeeze()  # Squeeze the input.
        outs = [layer(x) for layer in self.layers]  # Compute all task-head outputs for all layers.
        return torch.stack(outs, dim=-1)  # stacks all tensor into one tensor


class MultiTaskHead(nn.Module):
    """
    # Create task-head per task, direction.
    """

    def __init__(self, opts: argparse, transfer_learning_params: Union[list, None] = None):
        """
        Multi head task-head allocating for each task and direction a single task head.
        Args:
            opts: The model options.
            transfer_learning_params: list containing the associate taskhead params of the task, direction.
        """
        super(MultiTaskHead, self).__init__()
        self.opts = opts  # The options.
        self.ntasks = opts.ntasks  # The number of tasks.
        self.ndirections = opts.ndirections  # The number of directions.
        self.num_heads = opts.num_heads  # The number of heads.
        self.num_classes = opts.nclasses  # The number of classes for each task to create the head according to.
        self.taskhead = nn.ModuleList()  # TODO - CHANGE TO LIST OF LISTS.
        # For each task, direction create its task-head according to num_classes.
        for i in range(self.ntasks):
            for j in range(self.ndirections):
                # num_heads = self.num_heads[j]
                layer = HeadSingleTask(opts, self.num_classes[j], self.num_heads[j])  # create a taskhead.
                self.taskhead.append(layer)
                if transfer_learning_params is not None:
                    transfer_learning_params[i][j].extend(layer.parameters())  # Storing the taskhead params.

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Args:
            inputs: The output from BU2, and the flag.

        Returns: A tensor in the desired shape.

        """
        (bu2_out, flag) = inputs
        # In train mode we train only one head.
        if self.training or True:
            task_id = Flag_to_task(self.opts, flag)  # Get the task id.
            task_out = self.taskhead[task_id](bu2_out).squeeze()  # apply the appropriate task-head.
        # Otherwise, we test all heads and choose the desired by the direction flag.
        else:
            outputs = []  # All outputs list.
            for layer in self.taskhead:  # For each task head we compute the output.
                outputs.append(layer(bu2_out))
            direction_flag = flag[:, :self.ndirections]
            print(outputs[0].shape)

            outputs = torch.stack(outputs, dim=-1).squeeze()
            if len(outputs.shape) == 3:
                direction_flag = direction_flag.unsqueeze(dim=1)
            else:
                direction_flag = direction_flag.unsqueeze(dim=1)
                direction_flag = direction_flag.unsqueeze(dim=1)
            outputs = outputs * direction_flag  # Multiply by the direction flag mask.
            task_out = outputs.sum(dim=-1)  # Sum to have single prediction per task.

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
        filters = opts.nclasses[0]  # The number of binary classifiers needed to recognize all characters.
        infilters = opts.nfilters[-1]  # Output shape from the end of the BU1 stream.
        self.occurrence_transform = nn.Linear(infilters, filters)  # The linear transformation.

    def forward(self, inputs: torch) -> torch:
        """
        Args:
            inputs: The BU1 output.

        Returns: The binary classification input.

        """
        x = self.occurrence_transform(inputs)  # Apply the transform for the BU1 loss.
        return x
