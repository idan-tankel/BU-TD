"""
Here we define the heads, including single head, multi head and occurrence head.
"""
import argparse
from typing import Union

import torch
import torch.nn as nn
from torch import Tensor

from Data_Creation.Create_dataset_classes import DsType  # Import the Data_Creation set types.
from training.Data.Data_params import Flag
from training.Utils import Flag_to_task, create_single_one_hot


# Here we define the list_task_structs-head modules.

class HeadSingleTask(nn.Module):
    """
    Task head for single list_task_structs.
    Allocate tasks according to the desired output size.
    if model_test flag is NOFLAG we allocate subhead for each character.
    """

    def __init__(self, opts: argparse, nclasses: int, num_heads: int = 1):
        """
        Args:
            opts: The model_test options.
            nclasses: The number of classes.
            num_heads: Number of heads for the list_task_structs.
        """
        super(HeadSingleTask, self).__init__()
        self.opts = opts
        num_heads = nclasses if opts.model_flag is Flag.NOFLAG \
            else num_heads  # If The model_test flag is NOFLAG we allocate for each character a head o.w. according to the
        # nclasses.
        infilters = opts.nfilters[-1]  # The input size from the end of the BU2 stream.
        self.layers = nn.ModuleList(
            [nn.Linear(in_features=infilters, out_features=nclasses + 1) for _ in range(num_heads)])

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Args:
            inputs: The output from the model_test.

        Returns: A tensor with the shape of the number of classes.

        """
        x = inputs.squeeze()  # Squeeze the input.
        outs = [layer(x) for layer in self.layers]  # Compute all list_task_structs-head outputs for all layers.
        return torch.stack(tensors=outs, dim=-1)  # stacks all tensor into one tensor


class MultiTaskHead(nn.Module):
    """
    # Create list_task_structs-head per list_task_structs, list_task_structs.
    """

    def __init__(self, opts: argparse, transfer_learning_params: Union[list, None] = None):
        """
        Multi head list_task_structs-head allocating for each list_task_structs and list_task_structs a single list_task_structs head.
        Args:
            opts: The model_test options.
            transfer_learning_params: list containing the associate taskhead params of the list_task_structs, list_task_structs.
        """
        super(MultiTaskHead, self).__init__()
        self.opts = opts  # The options.
        self.ntasks = opts.ntasks  # The number of tasks.
        self.ndirections = opts.ndirections  # The number of directions.
        self.num_heads = opts.num_heads  # The number of heads.
        self.num_classes = opts.nclasses  # The number of classes for each list_task_structs to create the head according to.
        self.ds_type = self.opts.ds_type  # The data-set type.
        self.taskhead = nn.ModuleList()
        # For each list_task_structs, list_task_structs create its list_task_structs-head according to num_classes.
        for i in range(self.ntasks):
            for j in range(self.ndirections):
                layer = HeadSingleTask(opts=opts, nclasses=self.num_classes[i],
                                       num_heads=self.num_heads[j])  # create a taskhead.
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
        if self.training or self.ds_type is DsType.Omniglot:
            task_id = Flag_to_task(opts=self.opts, flags=flag)  # Get the list_task_structs id.
            task_out = self.taskhead[task_id](bu2_out)  # apply the appropriate list_task_structs-head.
        # Otherwise, we test all heads and choose the desired by the list_task_structs flag.
        else:
            outputs = []  # All outputs list.
            for layer in self.taskhead:  # For each list_task_structs head we compute the output.
                layer_out = layer(bu2_out)
                outputs.append(layer_out)
            outs = torch.stack(tensors=outputs, dim=1)  # Sum to have single prediction per list_task_structs.
            task_flag = create_single_one_hot(opts=self.opts, flags=flag)
            task_out = torch.einsum('ijkl,ij->ikl', outs, task_flag)  # Multiply the flag and the output.
        task_out = task_out.squeeze()
        return task_out


class OccurrenceHead(nn.Module):
    """
    Occurrence head, transforming the BU1 feature to binary classification over all classes.
    """

    def __init__(self, opts: argparse):
        """
        Occurrence head predicting for each character whether it exists in the sample.
        Args:
            opts: The model_test options.
        """
        super(OccurrenceHead, self).__init__()
        filters = opts.nclasses[0]  # The number of binary classifiers needed to recognize all characters.
        infilters = opts.nfilters[-1]  # Output shape from the end of the BU1 stream.
        self.occurrence_transform = nn.Linear(in_features=infilters, out_features=filters)  # The linear transformation.

    def forward(self, bu_out: Tensor) -> Tensor:
        """
        Args:
            bu_out: The BU1 output.

        Returns: The binary classification input.

        """
        x = self.occurrence_transform(input=bu_out)  # Apply the transform for the BU1 loss.
        return x
