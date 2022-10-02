import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from supp.general_functions import get_model_outs
from supp.data_functions import dev, preprocess, CE


# Accuracy#
def multi_label_accuracy_base(outs: object, samples: object) -> tuple:
    """
    The base class for all modes.
    Here for each head we compute its accuracy according to the model out and label task.
    Args:
        outs: The model outs.
        samples: The samples.

    Returns: The predictions, the task accuracy.

    """
    num_outputs = samples.label_task.shape[-1]
    cur_batch_size = samples.image.shape[0]
    predictions = torch.zeros((cur_batch_size, num_outputs), dtype=torch.int).to(dev, non_blocking=True)
    for k in range(num_outputs):
        task_output = outs.task[:, k, :]  # For each task extract its predictions.
        task_pred = torch.argmax(task_output, dim=1)  # Find the highest probability in the distribution
        predictions[:, k] = task_pred  # assign for each task its predictions
    label_task = samples.label_task
    task_accuracy = (  predictions == label_task).float()  # Compare the number of matches and normalize by the batch size*num_outputs.
    return predictions, task_accuracy  # return the predictions and the accuracy.


def multi_label_accuracy(outs: object, samples: object):
    """
    return the task accuracy mean over all samples.
    Args:
        outs: The model outs.
        samples: The samples.

    Returns: The predictions and task accuracy over the batch.

    """
    preds, task_accuracy = multi_label_accuracy_base(outs, samples)
    task_accuracy = task_accuracy.mean(axis=1)  # per single example
    return preds, task_accuracy


def multi_label_accuracy_weighted(outs, samples):
    """
    return the task accuracy weighted mean over the existing characters in the image.
    Args:
        outs: The model outs.
        samples: The samples.

    Returns: The predication and mean accuracy over the batch.

    """
    preds, task_accuracy = multi_label_accuracy_base(outs, samples)
    loss_weight = samples.loss_weight
    task_accuracy = task_accuracy * loss_weight
    task_accuracy = task_accuracy.sum(axis=1) / loss_weight.sum(axis=1)  # per single example
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
    loss_tasks = torch.zeros(samples.label_task.shape).to(dev, non_blocking=True)
    num_outputs = samples.label_task.shape[-1]
    for k in range(num_outputs):
        task_output = outs.task[:, k, :]  # For each task extract its last layer.
        label_task = samples.label_task[:, k]  # The label for the loss
        loss_task = CE(task_output, label_task)  # compute the loss
        loss_tasks[:, k] = loss_task  # Assign for each task its loss.
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
    loss_weight = samples.loss_weight
    losses_task = losses_task * loss_weight
    loss_task = losses_task.sum() / loss_weight.sum()  # a single valued result for the whole batch
    return loss_task


class UnifiedLossFun:
    def __init__(self, opts: argparse):
        """
        Args:
            opts: The model options
        """

        self.use_bu1_loss = opts.use_bu1_loss
        self.use_bu2_loss = opts.use_bu2_loss
        self.bu1_loss = opts.bu1_loss
        self.bu2_classification_loss = opts.bu2_loss
        self.inputs_to_struct = opts.inputs_to_struct
        self.opts = opts
        self.use_reg = opts.use_reg

    def __call__(self, model: nn.Module, inputs: list[torch], outs: list[torch]) -> float:
        """
        Args:
            model: The model.
            inputs: The input to the model.
            outs: The model outs.

        Returns: The overall loss.

        """
        outs = get_model_outs(model, outs)  # The output from all the streams.
        samples = self.inputs_to_struct(inputs)  # Make samples from the raw data.
        loss = 0.0  # The general loss.
        if self.use_bu1_loss:
            loss_occ = self.bu1_loss(outs.occurence_out,
                                     samples.label_existence)  # compute the binary existence classification loss
            loss += loss_occ  # Add the occurrence loss.                                
        if self.use_bu2_loss:
            loss_task = self.bu2_classification_loss(outs, samples)  # Compute the BU2 loss.
            loss += loss_task
        if self.use_reg:
            loss += self.opts.reg.loss_step()
        return loss


def accuracy(model: nn.Module, test_data_loader: DataLoader) -> float:
    """
    Args:
        model: The model options.
        test_data_loader: The test data.

    Returns: The accuracy over the batch size.

    """
    num_correct_pred, num_samples = (0.0, 0.0)
    for inputs in test_data_loader:  # Running over all inputs
        inputs = preprocess(inputs)  # Move to the cuda.
        num_samples += len(inputs[0])  # Update the number of samples.
        samples = model.module.inputs_to_struct(inputs)  # Make it struct.
        model.eval()  #
        outs = model(inputs)  # Compute the output.
        outs = get_model_outs(model, outs)  # From output to struct
        (_, task_accuracy_batch) = multi_label_accuracy_base(outs, samples)  # Compute the accuracy on the batch
        num_correct_pred += task_accuracy_batch.sum()  # Sum all accuracies on the batches.
    return num_correct_pred / num_samples  # Compute the mean.
