import argparse

import torch
import torch.nn as nn

from training.Data.Data_params import Flag
from training.Utils import tuple_direction_to_index


# Batch Norm class.

class BatchNorm(nn.Module):
    def __init__(self, opts: argparse, num_channels: int, dims: int = 2):
        """
        Creates batch_norm class.
        As continual learning overrides the running statistics, we created a class saving the running stats
        for each direction, task.
        We support storing and loading running stats of the model to dynamically evaluate the learning of the task.
        For each task, direction stores its mean,var as continual learning overrides those variables.
        Args:
            opts: The model options
            num_channels: num channels to apply batch_norm on.
            dims: apply 2d or 1d batch normalization.
        """
        super(BatchNorm, self).__init__()
        self.opts = opts
        self.ndirections = opts.ndirections
        self.ntasks = opts.ntasks
        self.save_stats = opts.model_flag is Flag.CL  # Save only for the continual learning flag.
        # Creates the norm function.
        if dims == 2:
            self.norm = nn.BatchNorm2d(num_channels)  # Create 2d BN.
        else:
            self.norm = nn.BatchNorm1d(num_channels)  # Create 1d BN.
        # creates list that should store the mean, var for each task and direction.
        if self.save_stats:
            # The running mean.
            self.running_mean_list = torch.zeros(opts.ndirections * opts.ntasks, num_channels)
            # The running variance.
            self.running_var_list = torch.zeros(opts.ndirections * opts.ntasks, num_channels)
            # Save the mean, variance.
            self.register_buffer("running_mean",
                                 self.running_mean_list)  # registering to the buffer to make it part of the meta-data.
            self.register_buffer("running_var",
                                 self.running_var_list)  # registering to the buffer to make it part of the meta-data.

    def forward(self, inputs: torch) -> torch:
        """
        # Applying the norm function on the input.
        Args:
            inputs: Tensor of dim [B,C,H,W].

        Returns: Tensor of dim [B,C,H,W].

        """
        return self.norm(inputs)  # applying the norm function.

    def load_running_stats(self, task_id: int, direction_tuple: tuple[int, int]) -> None:
        """
        Loads the mean, variance associated with the task_id and the direction_id.
        Args:
            task_id: The task id.
            direction_tuple: The direction tuple.

        """
        _, task_and_direction_idx = tuple_direction_to_index(self.opts.num_x_axis, self.opts.num_y_axis,
                                                             direction_tuple, self.opts.ndirections, task_id)
        running_mean = self.running_mean[task_and_direction_idx, :].detach().clone()  # Copy the running mean.
        running_var = self.running_var[task_and_direction_idx, :].detach().clone()  # Copy the running var.
        self.norm.running_mean = running_mean  # Assign the running mean.
        self.norm.running_var = running_var  # Assign the running var.

    def store_running_stats(self, task_id: int, direction_tuple: tuple) -> None:
        """
        Stores the mean, variance to the running_mean, running_var in the training time.
        Args:
            task_id: The task id.
            direction_tuple: The direction tuple.

        """
        # Get the index to load from.
        _, task_and_direction_idx = tuple_direction_to_index(self.opts.num_x_axis, self.opts.num_y_axis,
                                                             direction_tuple, self.opts.ndirections, task_id)
        running_mean = self.norm.running_mean.detach().clone()  # Get the index to load from.
        running_var = self.norm.running_var.detach().clone()  # Copy the running var.
        self.running_mean[task_and_direction_idx, :] = running_mean  # Store the running mean.
        self.running_var[task_and_direction_idx, :] = running_var  # Store the running var.


def store_running_stats(model: nn.Module, task_id: int, direction_id: tuple) -> None:
    """
    Stores the running_stats of the task_id for each norm_layer.
    Args:
        model: The model.
        task_id: The task id.
        direction_id: The direction id.

    """
    for _, layer in model.named_modules():  # Iterating over all layers.
        if isinstance(layer, BatchNorm):  # Save only for BatchNorm layers
            layer.store_running_stats(task_id, direction_id)  # For each BatchNorm instance store its running stats.


def load_running_stats(model: nn.Module, task_id: int, direction_id: tuple) -> None:
    """
    Loads the running_stats of the task_id for each norm_layer.
    Args:
        model: The model.
        task_id: The task id.
        direction_id: The direction id.

    """
    for _, layer in model.named_modules():  # Iterating over all layers.
        if isinstance(layer, BatchNorm):
            # For each BatchNorm instance load its running stats.
            layer.load_running_stats(task_id, direction_id)
