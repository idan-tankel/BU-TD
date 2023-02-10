import torch

from common.onTheRunInfo import get_loss


def loss_task_multi_label(taskk_out: torch.Tensor, label_taskk: torch.Tensor, weights_losses: torch.Tensor):
    # print("taskk_out: ", taskk_out.shape, "label_taskk: ", label_taskk.shape)
    # print("taskk_out max: ", taskk_out.max(), "label_taskk max: ", label_taskk.max())
    # print(weights_losses.dtype, taskk_out.dtype, label_taskk.dtype)
    # print("label_taskk: ", label_taskk, "\ntaskk_out: ", taskk_out.max())
    loss_function = get_loss(weights_losses)
    return loss_function(taskk_out, label_taskk)
    # OnTheRunInfo.loss.register_parameter('weight', weights_losses)
    # return OnTheRunInfo.loss(taskk_out, label_taskk)

# def losses(labels, outs: Tensor, number_classes: int):
#     """
#     losses function - calculate loss - for each image - then mean over images
#
#
#     :param outs: output of the model - shape (batch_size, num_classes)
#     :param labels: label of the task - shape (batch_size, num_classes)
#     :param number_classes: number of classes
#     """
#     losses_task = torch.zeros((labels.shape[0], number_classes)).to(OnTheRunInfo.dev,
#                                                                     non_blocking=True)
#
#     # Calculate right of
#     labels_task = get_labels_task_right_of(labels, number_classes)
#
#     weights_losses = torch.histc(labels_task, bins=number_classes + 1).to(torch.float32)
#     weights_losses[weights_losses == 0] = 1.0
#     weights_losses = 1 / weights_losses
#     for k in range(number_classes):
#         start_task_index = int(k * (number_classes + 1))
#         end_task_index = int(start_task_index + (number_classes + 1))
#         taskk_out = outs[:, start_task_index:end_task_index]
#         label_taskk = labels_task[:, k]
#         loss_taskk = loss_task_multi_label(taskk_out, label_taskk, weights_losses)
#         losses_task[:, k] = loss_taskk
#
#     # Calculate the ratio of right answers to all the others'
#     # ratio = number_classes / labels.shape[-1]
#
#     # Ratio of right answers to all the others
#     ratio = number_classes - (labels.shape[-1] - 1)
#
#     # Give equal weight to loss where not equal to 'number_classes' but to real digit
#     # losses_task[labels_task != number_classes] *= ratio
#
#     return losses_task.mean()
#
#
#
