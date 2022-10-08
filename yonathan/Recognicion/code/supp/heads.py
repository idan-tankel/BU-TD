import torch
import argparse
import torch.nn as nn
from supp.general_functions import flag_to_task
from supp.FlagAt import DsType, Flag

class HeadSingleTask(nn.Module):
    # Single task head.
    # allocates tasks according to the desired output size.
    # If all characters must be recognized size > 1 o.w. only 1 head be used.
    def __init__(self, opts: argparse, nclasses: list) -> None:
        """
        Args:
            opts: The model options.
            nclasses: The number of classes.
        """
        super(HeadSingleTask, self).__init__()
        layers = []
        if opts.model_flag is Flag.NOFLAG:
            nheads = nclasses
        else:
            nheads = 1

        for k in range(nheads):  # according to  the output size we allocate the number of heads.if flag=NOFLAG all characters(usually 6) will be recognized,the loop will run 6 times.
            outfilters = nclasses + 1  # The desired number of classes according to the task.
            infilters = opts.nfilters[-1]  # The input size from the end of the BU2 stream.
            layers.append(nn.Linear(infilters, outfilters))
        self.layers = nn.ModuleList(layers)

    def forward(self, inputs: torch) -> torch:
        """
        Args:
            inputs: The output from the model.

        Returns: A tensor with the shape of the number of classes.

        """
        x = inputs
        outs = []
        x = x.squeeze()
        for layer in self.layers:
            y = layer(x)  # Transforms the shape according to the number of classes.
            outs.append(y)
        return torch.stack(outs, dim=-2)  # stacks all tensor into one tensor


class MultiTaskHead(nn.Module):
    def __init__(self, opts: argparse):
        """
        Multi head task-head supporting all needed tasks.
        Args:
            opts: The model options.
        """
        super(MultiTaskHead, self).__init__()
        self.taskhead = []
        self.ntasks = opts.ntasks
        self.model_flag = opts.model_flag
        self.ndirections = opts.ndirections
        self.ds_type = opts.ds_type
        self.num_classes = opts.nclasses  # num_classes to create the task-heads according to.
        for i in range(self.ntasks * self.ndirections):  # For each task create its task-head according to num_clases.
            index = i // self.ndirections
            self.taskhead.append(HeadSingleTask(opts, self.num_classes[index]))
        self.taskhead = nn.ModuleList(self.taskhead)

    def forward(self, inputs: torch) -> torch:
        """
        Args:
            inputs: The output from BU2, the flag.

        Returns: A tensor in the desired shape.

        """
        (bu2_out, flag) = inputs
        if self.ds_type is DsType.Omniglot:
            direction_flag = flag[:, :self.ndirections]  # The task vector.
            lan_flag = flag[:, self.ndirections:self.ndirections + self.ntasks]
            direction_id = flag_to_task(direction_flag)
            lan_id = flag_to_task(lan_flag)
            idx = direction_id + self.ndirections * lan_id
        else:
            direction_flag = flag[:, :self.ndirections]  # The task vector.
            direction_id = flag_to_task(direction_flag)
            lan_id = 0
            idx = direction_id + self.ndirections * lan_id

        bu2_out = bu2_out.squeeze()  # Make it 1-dimensional.
        task_out = self.taskhead[idx](bu2_out)  # apply the appropriate task-head.
        if len(task_out.shape) == 2:
            task_out = task_out.unsqueeze(dim=2)
        return task_out


class OccurrenceHead(nn.Module):

    def __init__(self, opts: argparse):
        """
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
        x = self.occurrence_transform(x)
        return x
