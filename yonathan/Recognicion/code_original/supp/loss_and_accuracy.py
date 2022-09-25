import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from supp.measurments import get_model_outs
from supp.data_functions import dev, preprocess

# TODO-change to support the NOFLAG MODE.

# Accuracy#
def multi_label_accuracy_base(outs: object, samples: object, num_outputs: int = 1) -> tuple:
    """
    Args:
        outs: The model outs.
        samples: The samples.
        num_outputs: The number of output needed.

    Returns: The predictions, the task accuracy.

    """
    cur_batch_size = samples.image.shape[0]
    predictions = torch.zeros((cur_batch_size, num_outputs), dtype=torch.int).to(dev, non_blocking=True)
    for k in range(num_outputs):
        task_output = outs.task[:, :, k]  # For each task extract its predictions.
        task_pred = torch.argmax(task_output, axis=1)  # Find the highest probability in the distribution
        predictions[:, k] = task_pred  # assign for each task its predictions
    label_task = samples.label_task
    task_accuracy = (predictions == label_task).float() / (
        num_outputs)  # Compare the number of matches and normalize by the batch size*num_outputs.
    return predictions, task_accuracy  # return the predictions and the accuracy.

# Loss#

class BU2_loss:
    def __init__(self, num_outputs=1):
        """
        Args:
            num_outputs: The number of needed outputs.
        """

        self.num_outputs = num_outputs
        self.classification_loss = nn.CrossEntropyLoss(reduction='none').to(dev)

    def __call__(self, outs: object, samples: object) -> torch:
        """
        Args:
            outs: The model outputs.
            samples: The samples.

        Returns: The BU2 loss.

        """
        loss_tasks = torch.zeros(samples.label_task.shape).to(dev, non_blocking=True)
        for k in range(self.num_outputs):
            task_output = outs.task[:, :, k]  # For each task extract its last layer.
            label_task = samples.label_task[:, k]  # The label for the loss
            loss_task = self.classification_loss(task_output, label_task)  # compute the loss
            loss_tasks[:, k] = loss_task  # Assign for each task its loss.
        return loss_tasks.mean()  # return the task loss
'''
def multi_label_loss(outs: object, samples: object) -> float:
    """

    Args:
        outs:
        samples:

    Returns: Outputs from all the model, including BU1,TD,BU2 outputs.

    """
    """
    :param outs:Outputs from all the model, including BU1,TD,BU2 outputs.
    :param samples: The samples we train in the current step.
    :return: The average loss on the batch.
    """
    losses_task = multi_label_loss_base()(outs, samples)  # list of the losses per sample.
    return losses_task.mean()  # The average loss
'''

class UnifiedLossFun:
    def __init__(self, opts: argparse) -> None:
        """
        Args:
            opts: The model options
        """
        """
     :param opts: tells which losses to use and the loss functions.
     """
        self.use_bu1_loss = opts.use_bu1_loss
        self.use_bu2_loss = opts.use_bu2_loss
        self.bu1_loss = opts.bu1_loss
        self.bu2_classification_loss = opts.bu2_loss
        self.inputs_to_struct = opts.model.module.inputs_to_struct

    def __call__(self, model, inputs: list[torch], outs: list) -> float:
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
            loss_occ = self.bu1_loss(outs.occurence_out, samples.label_existence)  # compute the binary existence classification loss             
            loss += loss_occ  # Add the occurrence loss.                                
        if self.use_bu2_loss:
            loss_task = self.bu2_classification_loss()(outs, samples)  # Compute the BU2 loss.
            loss += loss_task
        return loss

def accuracy(opts: nn.Module, test_data_loader: DataLoader) -> float:
    """
    Args:
        opts: The model options.
        test_data_loader: The test data.

    Returns: The accuracy over the batch size.

    """
    num_correct_pred, num_samples = (0.0, 0.0)
    for inputs in test_data_loader:  # Running over all inputs
        inputs = preprocess(inputs)  # Move to the cuda.
        num_samples += len(inputs[0])  # Update the number of samples.
        samples = opts.model.module.inputs_to_struct(inputs)  # Make it struct.
        opts.model.eval() #
        outs = opts.model(inputs)  # Compute the output.
        outs = get_model_outs(opts.model, outs)  # From output to struct
        (_, task_accuracy_batch) = multi_label_accuracy_base(outs, samples)  # Compute the accuracy on the batch
        num_correct_pred += task_accuracy_batch.sum()  # Sum all accuracies on the batches.
    return num_correct_pred / num_samples  # Compute the mean.