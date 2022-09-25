import torch.nn as nn
import torch


class BatchNorm(nn.Module):
    def __init__(self, num_channels, num_tasks, dims=2):
        """
         creates batch_norm class.
         For each task stores its mean,var as consecutive trainings override this variables.
        :param num_channels: num channels to apply batch_norm on.
        :param num_tasks: access mean,var for each task
        :param dims: apply 2d or 1d batch normalization.
        """
        super(BatchNorm, self).__init__()
        # Creates the norm function.
        if dims == 2:
            norm = nn.BatchNorm2d(num_channels)
        else:
            norm = nn.BatchNorm1d(num_channels)
        self.norm = norm
        # creates list that should store the mean, var for each task.
        self.running_mean_list = []
        self.running_var_list = []
        for i in range(num_tasks):
            self.running_mean_list.append(torch.zeros(1, num_channels))  # Initializes the mean.
            self.running_var_list.append(torch.ones(1, num_channels))  # Initialized the variance.

        self.running_mean_list = torch.cat(self.running_mean_list, dim=0)  # Concatenating all means.
        self.running_var_list = torch.cat(self.running_var_list, dim=0)  # Concatenating all variances.
        self.register_buffer("running_mean",
                             self.running_mean_list)  # registering to the buffer to make it part of the meta-data.
        self.register_buffer("running_var",
                             self.running_var_list)  # registering to the buffer to make it part of the meta-data.

    def forward(self, inputs):
        """

        :param inputs: tensor of dim [B,C,H,W] or [B,C,H].
        :return: tensor of dim [B,C,H,W] or [B,C,H] respectively.
        """
        # applies the norm function.
        return self.norm(inputs)

    def load_running_stats(self, task_emb_id: int) -> None:
        """

        :param task_emb_id:
        :return: loads the saved mean, variance to the running_mean,var in the test time.
        """
        running_mean = self.running_mean[task_emb_id, :]
        running_var = self.running_var[task_emb_id, :]
        self.norm.running_mean = running_mean
        self.norm.running_var = running_var

    def store_running_stats(self, task_emb_id: int) -> None:
        """
        :param task_emb_id:
        :return: saves the mean, variance to the running_mean,var in the training time.
        """
        running_mean = self.norm.running_mean.clone()
        running_var = self.norm.running_var.clone()
        self.running_mean[task_emb_id, :] = running_mean
        self.running_var[task_emb_id, :] = running_var

# TODO - MAKE IT A MODEL FUNCTION.
def store_running_stats(model: nn.Module, task_emb_id: int) -> None:
    """
    Stores the running_stats of the task_id for each norm_layer.

    :param model: BUTD model
    :param task_emb_id: the task_id
    :return:
    """
    for _, layer in model.named_modules():
        if isinstance(layer, BatchNorm):
            layer.store_running_stats(task_emb_id)


def load_running_stats(model: nn.Module, task_emb_id: int) -> None:
    """
    loads the running_stats of the task_id for each norm_layer.

    :param model: BUTD model
    :param task_emb_id: the task_id
    :return:
    """
    for _, layer in model.named_modules():
        if isinstance(layer, BatchNorm):
            layer.load_running_stats(task_emb_id)

