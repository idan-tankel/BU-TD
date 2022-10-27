import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from supp.utils import preprocess

CE = nn.CrossEntropyLoss(reduction='none')


# Accuracy

def multi_label_accuracy_base(outs: object, samples: object) -> tuple:
    """
    The base class for all modes.
    Here for each head we compute its accuracy according to the model out and label task.
    Args:
        outs: The model outs.
        samples: The samples.

    Returns: The predictions, the task accuracy.

    """
    predictions = torch.argmax(outs.classifier, dim=1)  # Find the prediction for each queried character.
    label_task = samples.label_task  # The label task.
    task_accuracy = (predictions == label_task).float()  # Compute the number of matches.
    return predictions, task_accuracy  # return the predictions and the accuracy.


def multi_label_accuracy(outs: object, samples: object):
    """
    Compute the task accuracy mean over all samples for the guided model.
    Args:
        outs: The model outs.
        samples: The samples.

    Returns: The predictions and task accuracy over the batch.

    """
    preds, task_accuracy = multi_label_accuracy_base(outs, samples)  # The accuracy and the prediction.
    task_accuracy = task_accuracy.mean(axis=0)  # Compute the mean accuracy over all batch.
    return preds, task_accuracy  # return the predictions and the accuracy.


#
def multi_label_accuracy_weighted(outs: object, inputs: object):
    """
    Compute the task accuracy weighted mean over the existing characters in the image.
    Args:
        outs: The model outs.
        inputs: The samples.

    Returns: The predication and mean accuracy over the batch.

    """
    preds, task_accuracy = multi_label_accuracy_base(outs, inputs)  # The accuracy and the prediction.
    loss_weight = inputs.label_existence  # The weight of the desired characters.
    task_accuracy = task_accuracy * loss_weight  # Compute the accuracy over only existing characters.
    task_accuracy = task_accuracy.sum() / loss_weight.sum()  # Compute the mean over all characters and samples in the batch.
    return preds, task_accuracy  # return the predictions and the accuracy.


# Loss

def multi_label_loss_base(outs: object, samples: object):
    """
    Here for each head we compute its loss according to the model out and label task.
    Args:
        outs: The model outs.
        samples: The samples.

    Returns: The loss over all heads.

    """
    loss_tasks = CE(outs.classifier, samples.label_task)  # Taking the classes torch and the label task.
    return loss_tasks  # return the task loss


def multi_label_loss(outs, samples):
    """
    The loss over all images in the batch.
    Args:
        outs: The model outputs.
        samples: The samples.

    Returns: The mean loss over all samples in the batch.

    """
    losses_task = multi_label_loss_base(outs, samples)  # Compute the loss.
    loss_task = losses_task.mean()  # Mean over all batch.
    return loss_task


def multi_label_loss_weighted(outs, samples):
    """
    The loss over all existing characters in the batch.
    Args:
        outs: The model outputs.
        samples: The samples.

    Returns: The weighted loss over all existing characters, for not existing the loss is zero.

    """
    losses_task = multi_label_loss_base(outs, samples)  # Compute the loss.
    loss_weight = samples.label_existence  # The loss weight for only existing characters.
    losses_task = losses_task * loss_weight  # Compute the loss only in the existing characters.
    loss_task = losses_task.sum() / loss_weight.sum()  # Mean the loss over all batch and characters.
    return loss_task


def UnifiedCriterion(opts: argparse, inputs: list[torch.Tensor], outs: list[torch.Tensor]):
    """
    The sum over BU1 loss(if desired) and BU2 loss.
    Args:
        opts (argparse): The model options.
        inputs (list[torch.Tensor]): The input to the model.
        outs (list[torch.Tensor]): The model out.

    Returns:
        The overall loss.
    """
    outs = opts.model.outs_to_struct(outs)  # The output from all the streams.
    samples = opts.model.inputs_to_struct(inputs)  # Make samples from the raw data.
    loss = 0.0  # The general loss.
    if opts.use_bu1_loss:
        loss_occ = opts.bu1_loss(outs.occurence_out,
                                 samples.label_existence)  # compute the binary existence classification loss
        loss += loss_occ  # Add the occurrence loss.

    if opts.use_bu2_loss:
        loss_task = opts.bu2_loss(outs, samples)  # Compute the BU2 loss.
        loss += loss_task

    return loss


def accuracy(parser: nn.Module, test_data_loader: DataLoader) -> float:
    """
    Compute the accuracy of the model overall test_data_loader.
    Args:
        parser: The model options.
        test_data_loader: The test data.

    Returns: The accuracy over the batch size.

    """
    model = parser.model
    model.eval()
    num_correct_pred = 0.0,
    for inputs in test_data_loader:  # Running over all inputs.
        inputs = preprocess(inputs, parser.device)  # Move to the cuda.
        samples = parser.model.inputs_to_struct(inputs)  # Make it a struct.
        outs = model(inputs)  # Compute the output.
        outs = model.outs_to_struct(outs)  # From output to struct.
        (_, task_accuracy_batch) = parser.task_accuracy(outs, samples)  # Compute the accuracy on the batch.
        num_correct_pred += task_accuracy_batch  # Sum all accuracies on the batches.
    return num_correct_pred / len(test_data_loader)  # Compute the mean.
