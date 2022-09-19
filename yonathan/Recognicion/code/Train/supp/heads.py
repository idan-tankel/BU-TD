import torch
import argparse
import torch.nn as nn
from supp.general_functions import flag_to_task

class HeadSingleTask(nn.Module):
    # Single task head.
    def __init__(self, opts: argparse, nclasses: list):
        """
        Args:
            opts: The model options.
            nclasses: decided the number of classes according to the task.
        """

        super(HeadSingleTask, self).__init__()
        layers = []
        for k in range(opts.nheads):  # according to  the output size we allocate the number of heads.if flag=NOFLAG all characters(usually 6) will be recognized,the loop will run 6 times.
            outfilters = nclasses + 1  # The desired number of classes according to the task.
            infilters = opts.nfilters[-1]  # The input size from the end of the BU2 stream.
            layers.append(nn.Linear(infilters, outfilters))
        self.layers = nn.ModuleList(layers)

    def forward(self, inputs: torch) -> torch:
        """
        Args:
            inputs: Tensor of shape [B,out_filters]

        Returns: Tensor of shape [B, in_filters].

        """
        x = inputs
        outs = []
        x = x.squeeze()
        for layer in self.layers:
            y = layer(x)  # Transforms the shape according to the number of classes.
            outs.append(y)
        outs = outs
        return torch.stack(outs, dim=-1)  # stacks all tensor into one tensor

class MultiTaskHead(nn.Module):
    def __init__(self, opts: argparse):
        """
        Args:
            opts: The model options.
        """

        super(MultiTaskHead, self).__init__()

        self.ntasks = opts.ntasks
        self.model_flag = opts.model_flag
        self.num_classes = opts.nclasses  # num_classes to create the task-heads according to.
        self.ndirections = opts.ndirections
        self.taskhead = [[] for _ in range(self.ntasks)]
        for i in range(self.ntasks):
            for j in range(self.ndirections):  # For each task create its task-head according to num_clases.
                self.taskhead[i].append(HeadSingleTask(opts, self.num_classes[i]))
            self.taskhead[i] = nn.ModuleList(self.taskhead[i])
        self.taskhead = nn.ModuleList(self.taskhead)

    def forward(self, inputs: torch) -> torch:
        """
        Args:
            inputs: The input for the Head of shape [B,opts.nchannels[-1]]

        Returns: Tensor of shape [B,nclasses]

        """

        (bu2_out, flag) = inputs
        lan_flag = flag[2]
        task = flag_to_task(lan_flag)  # #TODO- change flag_to_direction -> flag_to_task
        direction_flag = flag[3]
        direction_idx = flag_to_task(direction_flag)
        bu2_out = bu2_out.squeeze()  # Make it 1-dimensional.
        task_out = self.taskhead[task][direction_idx](bu2_out)  # apply the appropriate task-head.
        if len(task_out.shape) == 2:
            task_out = task_out.unsqueeze(dim = 2)
        return task_out


class OccurrenceHead(nn.Module):

    def __init__(self, opts):
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
            inputs: [B,nchannels[-1]]

        Returns: [B,nclasses_existence]

        """
        x = inputs.squeeze()
        x = self.occurrence_transform(x)
        return x
