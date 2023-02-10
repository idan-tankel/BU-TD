import numpy as np
import torch
from torch import nn

from v26.ConstantsBuTd import *
from v26.models.Measurements import get_model_outs

from persons.code.v26.ConstantsBuTd import get_dev

loss_task_multi_label = nn.CrossEntropyLoss(reduction='none').to(get_dev())
loss_occurrence = torch.nn.BCEWithLogitsLoss(reduction='mean').to(get_dev())
loss_seg = torch.nn.MSELoss(reduction='mean').to(get_dev())
loss_task_op = nn.CrossEntropyLoss(reduction='mean').to(get_dev())


def multi_label_loss_base(outs, samples, nclasses):  # TODO - this function is written badly
    if isinstance(outs.task, list):  # TODO - remove duplicate - only where needed put if
        losses_task = []
        num_persons_start_features = 6
        for curr_example in list(range(len(outs.task))):
            example_out_list = outs.task[curr_example]
            example_out = []
            for curr_layer in list(range(len(example_out_list))):
                example_out.append(example_out_list[curr_layer])
            example_out = torch.stack(example_out)
            example_present_tasks = torch.where(samples.flag[curr_example][num_persons_start_features:] == 1)[
                0].tolist()
            example_gt = samples.person_features[curr_example, example_present_tasks]
            loss_taskk = loss_task_multi_label(example_out, example_gt)
            losses_task.append(loss_taskk)
        # losses_task = []
        # for curr_layer in range(
        #         len(nclasses)):
        #     taskk_out = outs.task[curr_layer]
        #     num_persons_start_features = 6  # TODO - 6 - to will be the number of features
        #     is_in_curr_task = samples.flag[:, num_persons_start_features + curr_layer] == 1
        #     samples_has_this_task = torch.where(is_in_curr_task)[0]
        #
        #     # TODO need to find for each - in this head - what was *actually* this feature
        #     label_taskk = samples.person_features[samples_has_this_task][:,
        #                   curr_layer]  # here cur_layer - is the feature - we find for each in this task - who has this mission - what was the feature
        #     loss_taskk = loss_task_multi_label(taskk_out, label_taskk)
        #
        #     if loss_taskk.shape[0] > 0:
        #         # losses_task[:, curr_layer] = loss_taskk
        #         losses_task.append(loss_taskk)
    else:

        losses_task = torch.zeros((samples.label_task.shape)).to(dev, non_blocking=True)
        for curr_layer in range(
                len(nclasses)):  # TODO - this we need to change/modify - diff for only one mission - not all the options at once(to check if this specific mission - got what it needed...)
            taskk_out = outs.task[:, :, curr_layer]
            label_taskk = samples.label_task[:, curr_layer]
            loss_taskk = loss_task_multi_label(taskk_out, label_taskk)
            losses_task[:, curr_layer] = loss_taskk
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
    loss_task = torch.cat(losses_task).mean()
    # loss_weight_by_task = activated_tasks(losses_task.shape[1], samples.flag)
    # losses_task = losses_task * loss_weight * loss_weight_by_task
    # loss_task = losses_task.sum() / loss_weight.sum()  # a single valued result for the whole batch
    return loss_task


#
# # Only the one in the task that is relevant to the task
# def multi_label_loss_weighted_loss_only_one(outs, samples, nclasses):
#     losses_task = multi_label_loss_base(outs, samples, nclasses)
#     loss_weight = samples.loss_weight
#
#     loss_weight_by_task = activated_tasks(losses_task.shape[1], samples.flag)
#     losses_task = losses_task * loss_weight * loss_weight_by_task
#     loss_task = losses_task.sum() / loss_weight.sum()  # a single valued result for the whole batch
#     return loss_task


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
