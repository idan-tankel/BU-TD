import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.Data.Structs import inputs_to_struct, outs_to_struct
from training.Utils import preprocess


# Here we define our accuracy functions.

def multi_label_accuracy_base(samples: inputs_to_struct, outs: outs_to_struct) -> tuple:
    """
    The base function for multi, weighted versions.
    Here for each head we compute its accuracy according to the model out and label task.
    Args:
        samples: The samples.
        outs: The model outs.


    Returns: The predictions, the task accuracy.

    """
    predictions = torch.argmax(outs.classifier, dim=1)  # Find the prediction for each queried character.
    label_task = samples.label_task  # The label task.
    task_accuracy = (predictions == label_task).float()  # Compute the number of matches.
    return predictions, task_accuracy  # return the predictions and the accuracy.


def multi_label_accuracy(samples: inputs_to_struct, outs: outs_to_struct):
    """
    Compute the task Accuracy mean over all samples for the guided model.
    Args:
        samples: The samples.
        outs: The model outs.


    Returns: The predictions and task Accuracy over the batch.

    """
    preds, task_accuracy = multi_label_accuracy_base(samples, outs)  # The Accuracy and the prediction.
    task_accuracy = task_accuracy.mean()  # Compute the mean Accuracy over the batch.
    return preds, task_accuracy  # return the predictions and the accuracy.


def multi_label_accuracy_weighted(samples: inputs_to_struct, outs: outs_to_struct):
    """
    Compute the task Accuracy weighted mean over the existing characters in the image.
    Args:
        samples: The inputs samples.
        outs: The model outs.

    Returns: The predication and mean Accuracy over the batch.

    """
    preds, task_accuracy = multi_label_accuracy_base(samples, outs)  # The Accuracy and the prediction.
    loss_weight = samples.label_existence  # The weight of the desired characters.
    task_accuracy = task_accuracy * loss_weight  # Compute the accuracy over only existing characters.
    # Compute the mean over all characters and samples in the batch.
    task_accuracy = task_accuracy.sum() / loss_weight.sum()
    return preds, task_accuracy


def accuracy(parser: argparse, model: nn.Module, test_data_loader: DataLoader) -> float:
    """
    Compute the accuracy of the model overall test_data_loader.
    Args:
        parser: The model options.
        model: The model.
        test_data_loader: The test data.

    Returns: The Accuracy over the test loadr.

    """
    model.eval()
    model = model.cuda()
    num_correct_preds = 0.0
    for inputs in test_data_loader:  # Running over all inputs.
        inputs = preprocess(inputs, 'cuda')  # Move to the cuda.
        samples = parser.inputs_to_struct(inputs)  # Make it a struct.
        outs = model(samples)  # Compute the output.
        outs = parser.outs_to_struct(outs)  # From output to struct.
        (_, task_accuracy_batch) = parser.task_accuracy(samples, outs)  # Compute the Accuracy on the batch.
        num_correct_preds += task_accuracy_batch  # Sum all accuracies on the batches.
    return num_correct_preds / len(test_data_loader)  # Compute the mean.
