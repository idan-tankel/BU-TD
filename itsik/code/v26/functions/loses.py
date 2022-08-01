import numpy as np
from torch import nn

from v26.ConstantsBuTd import *
from v26.funcs import activated_tasks
from v26.models.Measurements import get_model_outs

loss_task_multi_label = nn.CrossEntropyLoss(reduction='none').to(dev)
loss_occurrence = torch.nn.BCEWithLogitsLoss(reduction='mean').to(dev)
loss_seg = torch.nn.MSELoss(reduction='mean').to(dev)
loss_task_op = nn.CrossEntropyLoss(reduction='mean').to(dev)


def multi_label_loss_base(outs, samples, nclasses):
    losses_task = torch.zeros((samples.label_task.shape)).to(dev, non_blocking=True)
    for k in range(
            len(nclasses)):  # TODO - this we need to change/modify - diff for only one mission - not all the options at once(to check if this specific mission - got what it needed...)
        taskk_out = outs.task[:, :, k]
        label_taskk = samples.label_task[:, k]
        loss_taskk = loss_task_multi_label(taskk_out, label_taskk)
        losses_task[:, k] = loss_taskk
    return losses_task


def multi_label_loss(outs, samples, nclasses):
    losses_task = multi_label_loss_base(outs, samples, nclasses)
    loss_task = losses_task.mean(
    )  # a single valued result for the whole batch
    return loss_task


def multi_label_loss_weighted_loss(outs, samples, nclasses):
    losses_task = multi_label_loss_base(outs, samples, nclasses)
    loss_weight = samples.loss_weight

    losses_task = losses_task * loss_weight
    loss_task = losses_task.sum() / loss_weight.sum(
    )  # a single valued result for the whole batch
    return loss_task


# Only the one in the task that is relevant to the task
def multi_label_loss_weighted_loss_only_one(outs, samples, nclasses):
    losses_task = multi_label_loss_base(outs, samples, nclasses)
    loss_weight = samples.loss_weight

    loss_weight_by_task = activated_tasks(losses_task.shape[1], samples.flag)
    losses_task = losses_task * loss_weight * loss_weight_by_task
    loss_task = losses_task.sum() / loss_weight.sum()  # a single valued result for the whole batch
    return loss_task


def loss_fun(inputs, outs):
    # nn.CrossEntropyLoss on GPU is not deterministic. However using CPU doesn't seem to help either...
    outs = get_model_outs(get_model(), outs)
    inputs_to_struct = get_inputs_to_struct()
    samples = inputs_to_struct(inputs)
    losses = []
    if get_model_opts().use_bu1_loss:
        loss_occ = loss_occurrence(outs.occurence, samples.label_existence)
        losses.append(loss_occ)

    if get_model_opts().use_td_loss:
        loss_seg_td = loss_seg(outs.td_head, samples.seg)
        loss_bu1_after_convergence = 1
        loss_td_after_convergence = 100
        ratio = loss_bu1_after_convergence / loss_td_after_convergence
        losses.append(ratio * loss_seg_td)

    if get_model_opts().use_bu2_loss:
        loss_task = get_model_opts().bu2_loss(outs, samples, get_model_opts().nclasses)
        losses.append(loss_task)

    loss = torch.sum(torch.stack(losses))
    return loss
