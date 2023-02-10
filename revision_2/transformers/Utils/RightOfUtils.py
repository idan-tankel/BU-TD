import torch

from common.common_functions import get_the_output_right_of


def get_labels_task_right_of(labels, number_classes):
    return get_labels_task_right_of_f(labels, number_classes, number_classes)


def get_labels_task_right_of_f(labels, number_classes: int, fill_of_classes: int):
    labels_task = torch.zeros((labels.shape[0], number_classes)).to(torch.int64).to(labels.device)
    labels_task += fill_of_classes  # TODO - make it more elegant
    labels_task.scatter_(1, labels[:, :-1].to(torch.int64), labels[:, 1:].to(labels_task.dtype))
    return labels_task


def multi_label_accuracy_base(outs, samples, number_classes: int):
    labels = samples.label_all.squeeze(1)
    preds = get_the_output_right_of(labels, number_classes, outs)

    # Calculate right of
    labels_task = get_labels_task_right_of(labels, number_classes)

    flag_to_send = -1
    only_left_labels = get_labels_task_right_of_f(labels, number_classes, flag_to_send)  # TODO make this faster

    task_accuracy = (preds == labels_task).float()
    only_left_accuracy = torch.reshape(task_accuracy[(only_left_labels != flag_to_send)],
                                       (only_left_labels.shape[0], 5))  # TODO -check if gives correct reshspping
    return preds, task_accuracy, only_left_accuracy


def multi_label_accuracy(outs, samples, nclasses):
    preds, task_accuracy, only_left_accuracy = multi_label_accuracy_base(outs, samples, nclasses)
    task_accuracy = task_accuracy.mean(axis=1)  # per single example
    only_left_accuracy = only_left_accuracy.mean(axis=1)

    return preds, task_accuracy, only_left_accuracy
