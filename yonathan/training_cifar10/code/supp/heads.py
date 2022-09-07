import torch.nn as nn
import torch
from supp.general_functions import *
from supp.FlagAt import *
from supp.cifar_dataset import *
import argparse


class HeadSingleTask(nn.Module):
    # Single task head.
    # allocates tasks according to the desired output size.
    # If all characters must be recognized size > 1 o.w. only 1 head be used.
    def __init__(self, opts: argparse, nclasses: list) -> None:
        """
        :param opts: decided the input channels.
        :param nclasses: decided the number of classes according to the task.
        """
        super(HeadSingleTask, self).__init__()
        layers = []
        for k in range(len(nclasses)):  # according to  the output size we allocate the number of heads.if flag=NOFLAG all characters(usually 6) will be recognized,the loop will run 6 times.
            outfilters = nclasses[k]    # The desired number of classes according to the task.
            infilters = opts.nfilters[-1]  # The input size from the end of the BU2 stream.
            layers.append(nn.Linear(infilters, outfilters))
        self.layers = nn.ModuleList(layers)

    def forward(self, inputs: torch) -> torch:
        """
        :param inputs: tensor of shape [B,out_filters]
        :return: tensor of shape [B,num_outputs,in_filters] where num_outputs is the number of outputs the model should return
        #TODO -query the number of outputs
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
        :param opts: opts that tells the sizes transform to.
        """
        super(MultiTaskHead, self).__init__()
        self.taskhead = []
        self.ntasks = opts.ntasks
        self.model_flag = opts.model_flag
        self.num_classes = opts.nclasses  # num_classes to create the task-heads according to.
        for i in range(self.ntasks):  # For each task create its task-head according to num_clases.
            self.taskhead.append(HeadSingleTask(opts, self.num_classes[i]))
        self.taskhead = nn.ModuleList(self.taskhead)

    def forward(self, inputs: torch) -> torch:
        """
        :param inputs: bu2_out one dimensional tensor from BU2,the flag that says which task-head to choose.
        :return: one dimensional tensor of shape according to the needed number of classes.
        """
        (bu2_out, flag) = inputs
        task = flag_to_task(flag)  # #TODO- change flag_to_direction -> flag_to_task
        bu2_out = bu2_out.squeeze()  # Make it 1-dimensional.
        task_out = self.taskhead[task](bu2_out)  # apply the appropriate task-head.
        return task_out


class OccurrenceHead(nn.Module):
    # TODO-change omniglot_dataset to return also nclasses_existence.
    def __init__(self, opts):
        """
        :param opts:
        """
        super(OccurrenceHead, self).__init__()
        filters = opts.nclasses_existence  # The number of binary classifiers needed to recognize all characters.
        infilters = opts.nfilters[-1]  # Output shape from the end of the BU1 stream.
        self.occurrence_transform = nn.Linear(infilters, filters)  # The linear transformation.

    def forward(self, inputs: torch) -> torch:
        """
        :param inputs: [B,infilters]
        :return:       [B,nclasses_existence]
        """
        x = inputs.squeeze()
        x = self.occurrence_transform(x)
        return x


class ImageHead(nn.Module):
    # Takes as input the last layer of the TD stream and transforms into the original image shape.
    # Usually not used,but was created for a segmentation loss at the end of the TD-stream.
    def __init__(self, opts: argparse) -> None:
        """
        :param opts:
        """
        super(ImageHead, self).__init__()
        image_planes = opts.inshape[0]  # Image's channels.
        upsample_size = opts.strides[0]  # The size to Upsample to.
        infilters = opts.nfilters[0]  # The input's channel size.
        self.conv = conv3x3up(infilters, image_planes, upsample_size)  # The Upsampling:conv2d and then upsample.

    def forward(self, inputs):
        """
        :param inputs: [B,C1,H1,W1]
        :return: [B,C2,H2,W2] ([C2,H2,W2] original image shape)
        """
        x = self.conv(inputs)  # Performs the Upsampling.
        return x
