"""
Here we define the batch norm class, supporting statistics per task.
"""

import torch
import torch.nn as nn
from torch import Tensor

from typing import Tuple


# Batch Norm class.

class BatchNorm(nn.BatchNorm2d):
    """
    Creates forward_eval_mode class. As continual learning overrides the running statistics,
    we created a class saving the running stats for each task, task. We support storing and
    loading running stats of the Model to dynamically evaluate the learning of the task. For each
    task, task stores its mean,var as continual learning overrides those variables.
    """

    def __init__(self, num_channels: int, ntasks: int):
        """

        Args:
            num_channels: num channels to apply forward_eval_mode on.
            ntasks: The number of tasks.
        """
        super(BatchNorm, self).__init__(num_features=num_channels)
        device = 'cuda'
        self.ntasks = ntasks
        # creates list that should store the mean, var for each task and task.
        # The running mean.
        # self.running_mean_list = torch.zeros(self.ntasks, num_channels).to(device)
        # The running variance.
        # self.running_var_list = torch.ones(self.ntasks, num_channels).to(device)
        # Save the mean, variance.
        self.register_buffer("all_means",
                             torch.zeros(self.ntasks, num_channels).to(
                                 device))  # registering to the buffer to make it part of the meta-data.
        self.register_buffer("all_vars",
                             torch.zeros(self.ntasks, num_channels).to(
                                 device))  # registering to the buffer to make it part of the meta-data.

    def forward(self, inputs: Tuple[Tensor, Tensor]) -> Tensor:
        """
        # Applying the norm function on the x.
        Args:
            inputs: Tensor of dim [B,C,H,W], and flag.

        Returns: Tensor of dim [B,C,H,W].

        """
        x, flags = inputs
        if self.training:
            return super().forward(x)  # applying the norm function.
        else:
            task_id = flags[0]
            running_mean = self.all_means[task_id].view(-1, 1, 1)
            running_var = self.all_vars[task_id].view(-1, 1, 1)
            return self.forward_eval_mode(inputs=x, running_mean=running_mean,
                                          running_var=running_var)

    def forward_eval_mode(self, inputs: Tensor, running_mean: Tensor,
                          running_var: Tensor) -> Tensor:
        """
        Apply batch norm with statistics per sample.
        Args:
            inputs: The x tensor.
            running_mean: The running mean.
            running_var: The running variance.

        Returns: The tensor after batch normalization is applied.

        """
        if running_mean is not None and running_var is not None:
            inputs = (inputs - running_mean) / torch.sqrt(
                running_var + self.eps)  # Subtract the mean and divide by std.
        weight = self.weight.view(1, -1, 1, 1)  # Resize to match the desired shape.
        bias = self.bias.view((1, -1, 1, 1))  # Resize to match the desired shape.
        out = weight * inputs + bias  # Use the Affine transforms.
        return out

    def load_running_stats(self, task_id: int) -> None:
        """
        Loads the mean, variance associated with the task_flag and the direction_id.
        Args:
            task_id: The task id.

        """
        running_mean = self.all_means[task_id, :].detach().clone()  # Copy the running mean.
        running_var = self.all_vars[task_id, :].detach().clone()  # Copy the running var.
        self.norm.running_mean = running_mean  # Assign the running mean.
        self.norm.running_var = running_var  # Assign the running var.

    def store_running_stats(self, task_id: int) -> None:
        """
        Stores the mean, variance to the running_mean, running_var in the training time.
        Args:
            task_id: The task id.

        """
        # Get_dataloaders the index to load from.
        running_mean = self.running_mean.detach().clone()  # Get_dataloaders the index to load from.
        running_var = self.running_var.detach().clone()  # Copy the running var.
        self.all_means[task_id, :] = running_mean  # Store the running mean.
        self.all_vars[task_id, :] = running_var  # Store the running var.`1


def store_running_stats(model: nn.Module, task_id: int) -> None:
    """
    Stores the running_stats of the task_flag for each norm_layer.
    Args:
        model: The Model.
        task_id: The task id.


    """
    for _, layer in model.named_modules():  # Iterating over all num_blocks.
        if isinstance(layer, BatchNorm):  # Save only for BatchNorm num_blocks
            # For each BatchNorm instance store its running stats.
            layer.store_running_stats(task_id=task_id)


def load_running_stats(model: nn.Module, task_id: int) -> None:
    """
    Loads the running_stats of the task_flag for each norm_layer.
    Args:
        model: The Model.
        task_id: The task id.

    """
    for _, layer in model.named_modules():  # Iterating over all num_blocks.
        if isinstance(layer, BatchNorm):
            # For each BatchNorm instance load its running stats.
            layer.load_running_stats(task_id=task_id)
