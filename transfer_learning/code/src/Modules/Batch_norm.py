"""
Here we define the batch norm class, supporting statistics per task.
"""

import torch
import torch.nn as nn
from torch import Tensor


# Batch Norm class.

class BatchNorm(nn.Module):
    """
    Creates batch_norm_with_statistics_per_sample class. As continual learning overrides the running statistics,
    we created a class saving the running stats for each task, task. We support storing and
    loading running stats of the model to dynamically evaluate the learning of the task. For each
    task, task stores its mean,var as continual learning overrides those variables.
    """

    def __init__(self, num_channels: int, ntasks: int, dims: int = 2):
        """

        Args:
            num_channels: num channels to apply batch_norm_with_statistics_per_sample on.
            dims: apply 2d or 1d batch normalization.
        """
        super(BatchNorm, self).__init__()
        device = 'cuda'
        self.ntasks = ntasks
        # Creates the norm function.
        if dims == 2:
            self.norm = nn.BatchNorm2d(num_features=num_channels)  # Create 2d BN.
        else:
            self.norm = nn.BatchNorm1d(num_features=num_channels)  # Create 1d BN.
        # creates list that should store the mean, var for each task and task.
        # The running mean.
        self.running_mean_list = torch.zeros(self.ntasks, num_channels).to(device)
        # The running variance.
        self.running_var_list = torch.ones(self.ntasks, num_channels).to(device)
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
        if self.training:
            return self.norm(input=inputs)  # applying the norm function.
        else:
            device = inputs.device
            self.running_mean_list = self.running_mean_list.to(device)
            self.running_var_list = self.running_var_list.to(device)
            running_mean = (flags @ self.running_mean_list).unsqueeze(dim=2).unsqueeze(dim=2)
            running_var = (flags @ self.running_var_list).unsqueeze(dim=2).unsqueeze(dim=2)
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
        out = weight * inputs + bias  # Use the Affine transforms.
        return out

    def load_running_stats(self, task_id: int) -> None:
        """
        Loads the mean, variance associated with the task_flag and the direction_id.
        Args:
            task_id: The task id.

        """
        running_mean = self.running_mean_list[task_id, :].detach().clone()  # Copy the running mean.
        running_var = self.running_var_list[task_id, :].detach().clone()  # Copy the running var.
        self.norm.running_mean = running_mean  # Assign the running mean.
        self.norm.running_var = running_var  # Assign the running var.

    def store_running_stats(self, task_id: int) -> None:
        """
        Stores the mean, variance to the running_mean, running_var in the training time.
        Args:
            task_id: The task id.

        """
        # Get_dataloaders the index to load from.
        running_mean = self.norm.running_mean.detach().clone()  # Get_dataloaders the index to load from.
        running_var = self.norm.running_var.detach().clone()  # Copy the running var.
        self.running_mean_list[task_id, :] = running_mean  # Store the running mean.
        self.running_var_list[task_id, :] = running_var  # Store the running var.`1


def store_running_stats(model: nn.Module, task_id: int) -> None:
    """
    Stores the running_stats of the task_flag for each norm_layer.
    Args:
        model: The model.
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
        model: The model.
        task_id: The task id.

    """
    for _, layer in model.named_modules():  # Iterating over all num_blocks.
        if isinstance(layer, BatchNorm):
            # For each BatchNorm instance load its running stats.
            layer.load_running_stats(task_id=task_id)
