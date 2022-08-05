import torch.nn as nn
import torch
from supp.general_functions import *
from supp.FlagAt import *
from supp.omniglot_dataset import *
import argparse


class HeadSingleTask(nn.Module):
    # Single task head.
    def __init__(self, opts: argparse, out_filters, num_heads) -> None:
        """
        Args:
            opts: The model options.
            out_filters: The number of classes.
            num_heads: The number of heads.
        """
        super(HeadSingleTask, self).__init__()
        layers = []
        infilters = opts.nfilters[-1]  # The input size from the end of the BU2 stream.
        for _ in range(num_heads):  # according to  the output size we allocate the number of heads.if flag=NOFLAG all characters(usually 6) will be recognized,the loop will run 6 times.
            layers.append(nn.Linear(infilters, out_filters))
        self.layers = nn.ModuleList(layers)

    def forward(self, inputs: torch) -> torch:
        """
        Args:
            inputs: Tensor of shape [infilters,out_filters]

        Returns: [B,nclasses,nheads]

        """
        x = inputs
        outs = []
        for layer in self.layers:
            y = layer(x)  # Transforms the shape according to the number of classes.
            outs.append(y)
        return torch.stack(outs, dim=-1)  # stacks all tensor into one tensor


class MultiTaskHead(nn.Module):
    #Multi Task head class, assigning for each task its all three cycles heads.
    def __init__(self, opts: argparse):
        """
        Args:
            opts: The model options.
        """
        super(MultiTaskHead, self).__init__()
        self.taskhead = []
        self.ntasks = opts.ntasks
        self.model_flag = opts.model_flag
        self.num_classes = opts.nclasses  # num_classes to create the task-heads according to.
        self.grit_size = opts.grit_size
        for i in range(self.ntasks):  # For each task create its task-head according to num_clases.
            Task_heads = []
            Task_heads.append(HeadSingleTask(opts, 224//self.grit_size, num_heads = 2))
            Task_heads.append(HeadSingleTask(opts, 224//self.grit_size, num_heads = 2))
            Task_heads.append(HeadSingleTask(opts, self.num_classes[i][0] + 1, num_heads=1))
            Task_heads = nn.ModuleList(Task_heads)
            self.taskhead.append(Task_heads)
        self.taskhead = nn.ModuleList(self.taskhead)

    def forward(self, inputs: torch) -> torch:
        """
        Args:
            inputs: The one dimensional tensor from the BU2, the flag telling which task we solve.

        Returns: One dimensional tensor of shape according to the needed number of classes.

        """
        (bu2_out, flag,stage) = inputs
        task = flag_to_task(flag)  # #TODO- change flag_to_direction -> flag_to_task
        bu2_out = bu2_out.squeeze()  # Make it 1-dimensional.
        task_out = self.taskhead[task][stage](bu2_out)  # apply the appropriate task-head.
        return task_out


class OccurrenceHead(nn.Module):
    def __init__(self, opts:argparse):
        """
        Args:
            opts: The model options.
        """
        super(OccurrenceHead, self).__init__()
        filters = opts.nclasses[0][0]  # The number of binary classifiers needed to recognize all characters.
        infilters = opts.nfilters[-1]  # Output shape from the end of the BU1 stream.
        self.occurrence_transform = nn.Linear(infilters, filters)  # The linear transformation.

    def forward(self, inputs: torch) -> torch:
        """
        Args:
            inputs: [B,infilters]

        Returns: [B,nclasses_existence]

        """
        x = inputs.squeeze()
        x = self.occurrence_transform(x)
        return x


class TDImageHead(nn.Module):
    # Takes as input the last layer of the TD stream and transforms into the original image shape.
    # Usually not used,but was created for a segmentation loss at the end of the TD-stream.
    def __init__(self, opts: argparse) -> None:
        """
        Args:
            opts: The model options.
        """
        super(ImageHead, self).__init__()
        image_planes = opts.inshape[0]  # Image's channels.
        upsample_size = opts.strides[0]  # The size to Upsample to.
        infilters = opts.nfilters[0]  # The input's channel size.
        self.conv = conv3x3up(infilters, image_planes, upsample_size)  # The Upsampling:conv2d and then upsample.

    def forward(self, inputs:torch):
        """
        Args:
            inputs: [B,C1,H1,W1]

        Returns: [B,C2,H2,W2] ([C2,H2,W2] original image shape)
        """

        x = self.conv(inputs)  # Performs the Upsampling.
        return x

