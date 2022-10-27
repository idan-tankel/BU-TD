import argparse

import torch
import torch.nn as nn


# TODO - ADD SUPPORT TO STORING RUNNING STATS.


class BatchNorm(nn.Module):
    def __init__(self, opts: argparse, num_channels: int, dims: int = 2):
        """
        creates batch_norm class.
        As continual learning overrides the running stats, we created a class saving the running stats for each task.
        We support storing and loading running stats of the model to dynamically train and test the learning of the task.
        For each task stores its mean,var as continual learning overrides this variables.
        Args:
            opts: The model options
            num_channels: num channels to apply batch_norm on.
            dims: apply 2d or 1d batch normalization.
        """
        super(BatchNorm, self).__init__()
        self.ndirections = opts.ndirections
        self.ntasks = opts.ntasks
        # Creates the norm function.
        if dims == 2:
            self.norm = nn.BatchNorm2d(num_channels)  # Create 2d BN.
        else:
            self.norm = nn.BatchNorm1d(num_channels)  # Create 1d BN.
        # creates list that should store the mean, var for each task and direction.
        self.running_mean_list = torch.zeros((opts.ndirections * opts.ntasks, num_channels))  # Concatenating all means.
        self.running_var_list = torch.zeros(
            (opts.ndirections * opts.ntasks, num_channels))  # Concatenating all variances.
        self.register_buffer("running_mean",
                             self.running_mean_list)  # registering to the buffer to make it part of the meta-data.
        self.register_buffer("running_var",
                             self.running_var_list)  # registering to the buffer to make it part of the meta-data.

    def forward(self, inputs: torch):
        """

        Args:
            inputs: Tensor of dim [B,C,H,W] or [B,C,H].

        Returns: Tensor of dim [B,C,H,W] or [B,C,H] respectively.

        """
        return self.norm(inputs)  # applying the norm function.

    def load_running_stats(self, task_id: int, direction_id: int) -> None:
        """
        Loads the mean, variance associated with the task_id and the direction_id.
        Args:
            task_id: The task id.
            direction_id: The direction id.

        """

        idx = direction_id + self.ndirections * task_id  # Get the index to load from.
        running_mean = self.running_mean[idx, :].detach().clone()  # Copy the running mean.
        running_var = self.running_var[idx, :].detach().clone()  # Copy the running var.
        self.norm.running_mean = running_mean  # Assign the running mean.
        self.norm.running_var = running_var  # Assign the running var.

    def store_running_stats(self, task_id: int, direction_id: int) -> None:
        """
        Saves the mean, variance to the running_mean,var in the training time.
        Args:
            task_id: The task id.
            direction_id: The direction id.

        """
        idx = direction_id + self.ndirections * task_id  # Get the index to load from.
        running_mean = self.running_mean[idx, :].detach().clone()  # Get the index to load from.
        running_var = self.running_var[idx, :].detach().clone()  # Copy the running var.
        self.running_mean[idx, :] = running_mean  # Store the running mean.
        self.running_var[idx, :] = running_var  # Store the running var.


def store_running_stats(model: nn.Module, task_id: int, direction_id: int) -> None:
    """
    Stores the running_stats of the task_id for each norm_layer.
    Args:
        model: The model.
        task_id: The task id.
        direction_id: The direction id.

    """
    for _, layer in model.named_modules():
        if isinstance(layer, BatchNorm):
            layer.store_running_stats(task_id, direction_id)  # For each BatchNorm instance store its running stats.


def load_running_stats(model: nn.Module, task_id: int, direction_id: int) -> None:
    """
    Loads the running_stats of the task_id for each norm_layer.
    Args:
        model: The model.
        task_id: The task id.
        direction_id: The direction id.

    """
    for _, layer in model.named_modules():
        if isinstance(layer, BatchNorm):
            layer.load_running_stats(task_id, direction_id)  # For each BatchNorm instance load its running stats.
