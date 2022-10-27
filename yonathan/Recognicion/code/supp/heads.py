import argparse

import torch
import torch.nn as nn

from supp.Dataset_and_model_type_specification import Flag
from supp.utils import flag_to_task


class HeadSingleTask(nn.Module):
    # Task head for single task
    # Allocates tasks according to the desired output size.
    # if model flag is NOFLAG we allocate subhead for each character.
    def __init__(self, opts: argparse, nclasses: int) -> None:
        """
        Args:
            opts: The model options.
            nclasses: The number of classes.
        """
        super(HeadSingleTask, self).__init__()
        nheads = nclasses if opts.model_flag is Flag.NOFLAG else 1  # If The model flag is NOFLAG we allocate for each character a head o.w. single head.
        outfilters = nclasses + 1  # The desired number of classes according to the task.
        infilters = opts.nfilters[-1]  # The input size from the end of the BU2 stream.
        layers = [nn.Linear(infilters, outfilters) for _ in range(nheads)]
        self.layers = nn.ModuleList(layers)

    def forward(self, inputs: torch) -> torch:
        """
        Args:
            inputs: The output from the model.

        Returns: A tensor with the shape of the number of classes.

        """
        x = inputs.squeeze()
        outs = [layer(x) for layer in self.layers]
        return torch.stack(outs, dim=-1)  # stacks all tensor into one tensor


class MultiTaskHead(nn.Module):
    def __init__(self, opts: argparse, transfer_learning_params: list = None):
        """
        Multi head task-head allocating for each task and direction a single task head.
        Args:
            opts: The model options.
            transfer_learning_params: list containing the associate taskhead params of the task, direction.
        """
        super(MultiTaskHead, self).__init__()
        self.ntasks = opts.ntasks  # The number of tasks.
        self.ndirections = opts.ndirections  # The number of directions.
        self.num_classes = opts.nclasses  # The number of classes for each task to create the head according to.
        self.taskhead = []
        for i in range(
                self.ntasks * self.ndirections):  # For each task, direction create its task-head according to num_clases.
            index = i // self.ndirections
            layer = HeadSingleTask(opts, self.num_classes[index])  # create a taskhead
            self.taskhead.append(layer)
            if transfer_learning_params is not None:
                transfer_learning_params[i].extend(layer.parameters())  # Storing the taskhead params.
        self.taskhead = nn.ModuleList(self.taskhead)

    def forward(self, inputs: torch, idx_out=None) -> torch:
        """
        Args:
            inputs: The output from BU2, the flag.
            idx_out: If online head modification is needed we support id idx_out is not none.

        Returns: A tensor in the desired shape.

        """
        (bu2_out, flag) = inputs
        direction_flag = flag[:, :self.ndirections]  # The direction one hot flag.
        task_flag = flag[:, self.ndirections:self.ndirections + self.ntasks]  # The task one hot flag.
        direction_id = flag_to_task(direction_flag)  # The direction id.
        task_id = flag_to_task(task_flag)  # The task id.
        idx = direction_id + self.ndirections * task_id if idx_out is None else idx_out  # The head index with possible modification according to idx_out.
        task_out = self.taskhead[idx](bu2_out).squeeze()  # apply the appropriate task-head.
        #   if len(task_out.shape) == 2:
        #       task_out = task_out.unsqueeze(dim=2) # Unsqueeze to match the shape needed for the loss/accuracy.
        return task_out


class OccurrenceHead(nn.Module):

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
        x = inputs.squeeze()
        x = self.occurrence_transform(x)  # Apply the transform for the BU1 loss.
        return x
