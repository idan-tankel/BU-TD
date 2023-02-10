import torch

from v26.ConstantsBuTd import *


def get_bounding_box(mask):
    if len(mask.shape) > 2:
        bool_mask = mask[:, :, 1]
    else:
        bool_mask = mask
    h, w = bool_mask.shape
    rows, cols = bool_mask.nonzero()
    margin = 5
    stx = min(cols)
    sty = min(rows)
    endx = max(cols)
    endy = max(rows)
    stx -= margin
    sty -= margin
    endx += margin
    endy += margin
    stx = max(0, stx)
    sty = max(0, sty)
    endx = min(endx, w)
    endy = min(endy, h)
    return [stx, sty, endx, endy]


def multi_label_accuracy_base(outs, samples, nclasses):  # TODO - this function is written badly
    cur_batch_size = samples.image.shape[0]
    if isinstance(outs.task, list):
        preds = []
    else:
        preds = torch.zeros((cur_batch_size, len(nclasses)), dtype=torch.int).to(dev, non_blocking=True)

    num_persons_start_features = 6  # TODO - 6 - to will be the number of features

    if isinstance(outs.task, list):
        for curr_example_index, curr_example in enumerate(outs.task):
            curr_example_pred = []
            for task_number, curr_task_in_example in enumerate(curr_example):

                if curr_task_in_example.shape[0] > 0:
                    predsk = curr_task_in_example.argmax()

                    # is_in_curr_task = samples.flag[:, num_persons_start_features] == 1
                    current_flag: int = \
                        torch.where(samples.flag[curr_example_index, num_persons_start_features:] == 1)[
                            task_number].item()

                    curr_example_pred.append(samples.person_features[curr_example_index][current_flag] == predsk)
            preds.append(torch.as_tensor(curr_example_pred))
    else:

        for curr_layer in range(len(nclasses)):
            taskk_out = outs.task[:, :, curr_layer]
            predsk = torch.argmax(taskk_out, axis=1)
            preds[:, curr_layer] = predsk
    if isinstance(outs.task, list):
        task_accuracy = preds
    else:
        label_task = samples.label_task
        task_accuracy = (preds == label_task).float()
    return preds, task_accuracy


def multi_label_accuracy(outs, samples, nclasses):
    preds, task_accuracy = multi_label_accuracy_base(outs, samples, nclasses)
    task_accuracy = task_accuracy.mean(axis=1)  # per single example
    return preds, task_accuracy


def multi_label_accuracy_weighted_loss(outs, samples, nclasses):
    preds, task_accuracy = multi_label_accuracy_base(outs, samples, nclasses)
    loss_weight = samples.loss_weight
    task_accuracy = task_accuracy * loss_weight
    task_accuracy = task_accuracy.sum(axis=1) / loss_weight.sum(
        axis=1)  # per single example
    return preds, task_accuracy


def multi_label_accuracy_weighted_loss_only_one(outs, samples, nclasses):
    preds, task_accuracy = multi_label_accuracy_base(outs, samples, nclasses)
    task_accuracy = torch.cat(task_accuracy).float().mean() * samples.flag.shape[0]
    return preds, task_accuracy
