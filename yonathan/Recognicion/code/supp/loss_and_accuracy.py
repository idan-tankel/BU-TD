import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from supp.general_functions import preprocess

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
    predictions = torch.argmax(outs.task,dim=1)
    label_task = samples.label_task
    task_accuracy = ( predictions == label_task).float()  # Compare the number of matches and normalize by the batch size*num_outputs.
    return (predictions, task_accuracy)  # return the predictions and the accuracy.

def multi_label_accuracy(outs: object, samples: object):
    """
    return the task accuracy mean over all samples.
    Args:
        outs: The model outs.
        samples: The samples.

    Returns: The predictions and task accuracy over the batch.

    """
    preds, task_accuracy = multi_label_accuracy_base(outs, samples)
    task_accuracy = task_accuracy.mean(axis=0)  # per single example
    return preds, task_accuracy
#
def multi_label_accuracy_weighted(outs, inputs):
    """
    return the task accuracy weighted mean over the existing characters in the image.
    Args:
        outs: The model outs.
        samples: The samples.

    Returns: The predication and mean accuracy over the batch.

    """
    preds, task_accuracy = multi_label_accuracy_base(outs, inputs)
    loss_weight = inputs.label_existence
    task_accuracy = task_accuracy * loss_weight
    task_accuracy =  task_accuracy.sum() / loss_weight.sum() # per single example
    return preds, task_accuracy

# Loss

def multi_label_loss_base(outs: object, samples: object):
    """
    Here for each head we compute its loss according to the model out and label task.
    Args:
        outs: The model outs.
        samples: The samples.

    Returns: The loss over all heads.

    """
    loss_tasks = CE(outs.task, samples.label_task)
    return loss_tasks  # return the task loss

def multi_label_loss(outs, samples):
    """
    The loss over all images in the batch.
    Args:
        outs: The model outputs.
        samples: The samples.

    Returns: The mean loss over all samples in the batch.

    """
    losses_task = multi_label_loss_base(outs, samples)
    loss_task = losses_task.mean()  # a single valued result for the whole batch
    return loss_task

def multi_label_loss_weighted(outs, samples):
    """
    The loss over all existing characters in the batch.
    Args:
        outs: The model outputs.
        samples: The samples.

    Returns: The weighted loss over all existing characters, for not existing the loss is zero.

    """
    losses_task = multi_label_loss_base(outs, samples)
    loss_weight = samples.label_existence
    losses_task = losses_task * loss_weight
    loss_task = losses_task.sum() / loss_weight.sum()  # a single valued result for the whole batch
    return loss_task

def UnifiedCriterion(opts:argparse, inputs: list[torch], outs: list[torch]):
    outs = opts.model.outs_to_struct(outs)  # The output from all the streams.
    samples = opts.inputs_to_struct(inputs)  # Make samples from the raw data.
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
    Args:
        model: The model options.
        test_data_loader: The test data.

    Returns: The accuracy over the batch size.

    """
    model = parser.model
    num_correct_pred, num_samples = (0.0, 0.0)
    for inputs in test_data_loader:  # Running over all inputs
        inputs = preprocess(parser, inputs)  # Move to the cuda.
        num_samples += 1  # Update the number of samples.
        samples = parser.inputs_to_struct(inputs)  # Make it struct.
        model.train()  #
        outs = model(inputs)  # Compute the output.
        outs = model.outs_to_struct(outs)  # From output to struct
        ( _ , task_accuracy_batch) = multi_label_accuracy_weighted(outs, samples)  # Compute the accuracy on the batch
        num_correct_pred += task_accuracy_batch.sum()  # Sum all accuracies on the batches.
    return num_correct_pred / num_samples  # Compute the mean.
