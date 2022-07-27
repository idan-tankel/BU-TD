import torch

from persons.code.v26.ConstantsBuTd import *


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


def multi_label_accuracy_base(outs, samples, nclasses):
    cur_batch_size = samples.image.shape[0]
    preds = torch.zeros((cur_batch_size, len(nclasses)),
                        dtype=torch.int).to(dev, non_blocking=True)
    for k in range(len(nclasses)):
        taskk_out = outs.task[:, :, k]
        predsk = torch.argmax(taskk_out, axis=1)
        preds[:, k] = predsk
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
