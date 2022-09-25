import torch.nn as nn
import torch

class BatchNorm(nn.Module):
    def __init__(self,opts, num_channels, dims=2):
        """
        creates batch_norm class.
        For each task stores its mean,var as consecutive trainings override this variables.
        Args:
            num_channels: num channels to apply batch_norm on.
            num_tasks: Number of tasks
            dims: apply 2d or 1d batch normalization.
        """
        super(BatchNorm, self).__init__()
        # Creates the norm function.
        if dims == 2:
            norm = nn.BatchNorm2d(num_channels)
        else:
            norm = nn.BatchNorm1d(num_channels)
        self.norm = norm
        # creates list that should store the mean, var for each task.
        self.running_mean_list =torch.zeros((opts.ndirections,opts.ntasks,num_channels))  # Concatenating all means.
        self.running_var_list = torch.zeros((opts.ndirections,opts.ntasks,num_channels))  # Concatenating all variances.
        self.register_buffer("running_mean", self.running_mean_list)  # registering to the buffer to make it part of the meta-data.
        self.register_buffer("running_var", self.running_var_list)  # registering to the buffer to make it part of the meta-data.

    def forward(self, inputs):
        """

        Args:
            inputs: Rensor of dim [B,C,H,W] or [B,C,H].

        Returns: Tensor of dim [B,C,H,W] or [B,C,H] respectively.

        """
        # applies the norm function.
        return self.norm(inputs)

    def load_running_stats(self, task_emb_id: int,direction:int) -> None:
        """
        Args:
            task_emb_id: The task_id.

        Returns: Loads the saved mean, variance to the running_mean,var in the test time.

        """
        running_mean = self.running_mean[direction,task_emb_id, :]
        running_var = self.running_var[direction,task_emb_id, :]
        self.norm.running_mean = running_mean
        self.norm.running_var = running_var

    def store_running_stats(self, task_emb_id: int,direction:int) -> None:
        """
        Args:
            task_emb_id: The task_id.

        Returns: Saves the mean, variance to the running_mean,var in the training time.

        """
        running_mean = self.norm.running_mean
        running_var = self.norm.running_var
        self.running_mean[direction,task_emb_id, :] = running_mean.clone()
        self.running_var[direction,task_emb_id, :] = running_var.clone()

# TODO - MAKE IT A MODEL FUNCTION.
def store_running_stats(model: nn.Module, task_emb_id: int,direction_id) -> None:
    """
    Stores the running_stats of the task_id for each norm_layer.
    Args:
        model: The model.
        task_emb_id: the task_id.

    """
    for _, layer in model.named_modules():
        if isinstance(layer, BatchNorm):
            layer.store_running_stats(task_emb_id, direction_id)


def load_running_stats(model: nn.Module, task_emb_id: int,direction_id) -> None:
    """
    loads the running_stats of the task_id for each norm_layer.
    Args:
        model: The model.
        task_emb_id: the task_id.

    """
    for _, layer in model.named_modules():
        if isinstance(layer, BatchNorm):
            layer.load_running_stats(task_emb_id, direction_id)

