"""
Here we define the batch norm class, supporting statistics per list_task_structs.
"""
import argparse

import torch
import torch.nn as nn
from torch import Tensor

from training.Data.Data_params import Flag
from training.Utils import tuple_direction_to_index, create_single_one_hot


# Batch Norm class.

class BatchNorm(nn.Module):
    """
    Creates batch_norm_with_statistics_per_sample class.
    As continual learning overrides the running statistics, we created a class saving the running stats
    for each list_task_structs, list_task_structs.
    We support storing and loading running stats of the model to dynamically evaluate the learning of the list_task_structs.
    For each list_task_structs, list_task_structs stores its mean,var as continual learning overrides those variables.
    """

    def __init__(self, opts: argparse, num_channels: int, dims: int = 2, device='cuda'):
        """

        Args:
            opts: The model options
            num_channels: num channels to apply batch_norm_with_statistics_per_sample on.
            dims: apply 2d or 1d batch normalization.
            device: The device.
        """
        super(BatchNorm, self).__init__()
        self.opts = opts
        self.ndirections = opts.ndirections
        self.ntasks = opts.ntasks
        self.save_stats = opts.model_flag is Flag.CL and opts.save_stats  # Save only for the continual learning flag.
        # Creates the norm function.
        if dims == 2:
            self.norm = nn.BatchNorm2d(num_features=num_channels)  # Create 2d BN.
        else:
            self.norm = nn.BatchNorm1d(num_features=num_channels)  # Create 1d BN.
        # creates list that should store the mean, var for each list_task_structs and list_task_structs.
        if self.save_stats:
            # The running mean.
            self.running_mean_list = torch.zeros(opts.ndirections * opts.ntasks, num_channels).to(device)
            # The running variance.
            self.running_var_list = torch.ones(opts.ndirections * opts.ntasks, num_channels).to(device)
            # Save the mean, variance.
            self.register_buffer("running_mean",
                                 self.running_mean_list)  # registering to the buffer to make it part of the meta-data.
            self.register_buffer("running_var",
                                 self.running_var_list)  # registering to the buffer to make it part of the meta-data.

    def forward(self, inputs: Tensor, flags: Tensor) -> Tensor:
        """
        # Applying the norm function on the input.
        Args:
            inputs: Tensor of dim [B,C,H,W].
            flags: The flag, needed for evaluation on different tasks, having each its own running statistics.

        Returns: Tensor of dim [B,C,H,W].

        """
        if self.training or not self.save_stats:
            return self.norm(input=inputs)  # applying the norm function.
        else:
            task_flag = create_single_one_hot(opts=self.opts, flags=flags)
            running_mean = (task_flag @ self.running_mean_list).unsqueeze(dim=2).unsqueeze(dim=2)
            running_var = (task_flag @ self.running_var_list).unsqueeze(dim=2).unsqueeze(dim=2)
            running_var = running_var if not self.training or self.track_running_stats else None
            running_mean = running_mean if not self.training or self.track_running_stats else None
            return self.batch_norm_with_statistics_per_sample(inputs=inputs, running_mean=running_mean,
                                                              running_var=running_var)

    def batch_norm_with_statistics_per_sample(self, inputs: Tensor, running_mean: Tensor,
                                              running_var: Tensor) -> Tensor:
        """
        Apply batch norm with statistics per sample.
        Args:
            inputs: The input tensor.
            running_mean: The running mean.
            running_var: The running variance.

        Returns: The tensor after batch normalization is applied.

        """
        if running_mean is not None and running_var is not None:
            inputs = (inputs - running_mean) / torch.sqrt(
                running_var + self.norm.eps)  # Subtract the mean and divide by std.
        weight = self.norm.weight.view(1, -1, 1, 1)  # Resize to match the desired shape.
        bias = self.norm.bias.view((1, -1, 1, 1))  # Resize to match the desired shape.
        out = weight * inputs + bias  # Use the Affine transform.
        return out

    def load_running_stats(self, task_id: int, direction_tuple: tuple[int, int]) -> None:
        """
        Loads the mean, variance associated with the task_id and the direction_id.
        Args:
            task_id: The list_task_structs id.
            direction_tuple: The list_task_structs tuple.

        """
        _, task_and_direction_idx = tuple_direction_to_index(num_x_axis=self.opts.num_x_axis,
                                                             num_y_axis=self.opts.num_y_axis,
                                                             direction=direction_tuple,
                                                             ndirections=self.opts.ndirections, task_id=task_id)
        running_mean = self.running_mean_list[task_and_direction_idx, :].detach().clone()  # Copy the running mean.
        running_var = self.running_var_list[task_and_direction_idx, :].detach().clone()  # Copy the running var.
        self.norm.running_mean = running_mean  # Assign the running mean.
        self.norm.running_var = running_var  # Assign the running var.

    def store_running_stats(self, task_id: int, direction_tuple: tuple[int, int]) -> None:
        """
        Stores the mean, variance to the running_mean, running_var in the training time.
        Args:
            task_id: The list_task_structs id.
            direction_tuple: The list_task_structs tuple.

        """
        # Get the index to load from.
        _, task_and_direction_idx = tuple_direction_to_index(num_x_axis=self.opts.num_x_axis,
                                                             num_y_axis=self.opts.num_y_axis,
                                                             direction=direction_tuple,
                                                             ndirections=self.opts.ndirections, task_id=task_id)
        running_mean = self.norm.running_mean.detach().clone()  # Get the index to load from.
        running_var = self.norm.running_var.detach().clone()  # Copy the running var.
        self.running_mean_list[task_and_direction_idx, :] = running_mean  # Store the running mean.
        self.running_var_list[task_and_direction_idx, :] = running_var  # Store the running var.`1


def store_running_stats(model: nn.Module, task_id: int, direction_id: tuple) -> None:
    """
    Stores the running_stats of the task_id for each norm_layer.
    Args:
        model: The model.
        task_id: The list_task_structs id.
        direction_id: The list_task_structs id.

    """
    for _, layer in model.named_modules():  # Iterating over all layers.
        if isinstance(layer, BatchNorm):  # Save only for BatchNorm layers
            # For each BatchNorm instance store its running stats.
            layer.store_running_stats(task_id=task_id,
                                      direction_tuple=direction_id)


def load_running_stats(model: nn.Module, task_id: int, direction_id: tuple) -> None:
    """
    Loads the running_stats of the task_id for each norm_layer.
    Args:
        model: The model.
        task_id: The list_task_structs id.
        direction_id: The list_task_structs id.

    """
    for _, layer in model.named_modules():  # Iterating over all layers.
        if isinstance(layer, BatchNorm):
            # For each BatchNorm instance load its running stats.
            layer.load_running_stats(task_id=task_id, direction_tuple=direction_id)
