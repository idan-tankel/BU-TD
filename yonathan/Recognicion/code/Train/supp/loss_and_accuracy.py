import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from supp.measurments import set_datasets_measurements, get_model_outs
from supp.data_functions import dev, preprocess

# TODO-change to support the NOFLAG mode.
def multi_label_accuracy_base(outs: object, samples: object, num_outputs: int = 1) -> tuple:
    """
    :param outs:Outputs from all the model, including BU1,TD,BU2 outputs.
    :param samples: The samples we train in the current step.
    :param num_outputs: The number outputs the model should return(usually 1 or the number of characters in the image).
    :return: The predictions and the accuracy on the batch.
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
# TODO-change to support the NOFLAG MODE.
# loss_task_multi_label = nn.CrossEntropyLoss(reduction='none').to(dev)
class multi_label_loss_base:
    def __init__(self, num_outputs=1):
        """
        :param parser:
        :param num_outputs:
        """
        self.num_outputs = num_outputs
        self.classification_loss = nn.CrossEntropyLoss(reduction='none').to(dev)

    def __call__(self, outs: object, samples: object) -> torch:
        """
         :param outs:Outputs from all the model, including BU1,TD,BU2 outputs.
         :param samples: The samples we train in the current step.
         :param num_outputs: The number outputs the model should return(usually 1 or the number of characters in the image).
         :return:The loss on the batch.
         """
        loss_tasks = torch.zeros(samples.label_task.shape).to(dev, non_blocking=True)
        for k in range(self.num_outputs):
            task_output = outs.task[:, :, k]  # For each task extract its last layer.
            label_task = samples.label_task[:, k]  # The label for the loss
            loss_task = self.classification_loss(task_output, label_task)  # compute the loss
            loss_tasks[:, k] = loss_task  # Assign for each task its loss.
        return loss_tasks  # return the task loss


def multi_label_loss(outs: object, samples: object) -> float:
    """
    :param outs:Outputs from all the model, including BU1,TD,BU2 outputs.
    :param samples: The samples we train in the current step.
    :return: The average loss on the batch.
    """
    losses_task = multi_label_loss_base()(outs, samples)  # list of the losses per sample.
    return losses_task.mean()  # The average loss


class UnifiedLossFun:
    def __init__(self, opts: argparse) -> None:
        """
     :param opts: tells which losses to use and the loss functions.
     """
        self.use_bu1_loss = opts.use_bu1_loss
        self.use_bu2_loss = opts.use_bu2_loss
        self.bu1_loss = opts.bu1_loss
        self.td_loss = opts.td_loss
        self.bu2_classification_loss = opts.bu2_loss
        self.inputs_to_struct = opts.model.module.inputs_to_struct
        self.opts = opts
     #   self.regulizer = opts.regulizer

    def __call__(self,opts, inputs: list[torch], outs: list) -> float:
        """
        :param inputs: Input to the model.
        :param outs: Output from the model.
        :return: The combined loss over all the stream.
        """
        outs = get_model_outs(self.opts.model, outs)  # The output from all the streams.
        samples = self.inputs_to_struct(inputs)  # Make samples from the raw data.
        loss = 0  # The general loss.
        if self.use_bu1_loss:
            loss_occ = self.bu1_loss(outs.occurence_out, samples.label_existence)  # compute the binary existence classification loss
            loss += loss_occ  # Add the occurrence loss.

        if self.use_bu2_loss:
            loss_task = self.bu2_classification_loss(outs, samples)  # Compute the BU2 loss.
            loss += loss_task
        if opts.use_reg:
         loss_new = opts.reg.regulizer.ewc_loss()
         loss += loss_new
        return loss


def accuracy(model: nn.Module, test_data_loader: DataLoader) -> float:
    """
    :param opts:The model options to compute its accuracy.
    :param test_data_loader: The data.
    :return:The computed accuracy.
    """
    num_correct_pred, num_samples = (0.0, 0.0)

    for inputs in test_data_loader:  # Running over all inputs
        inputs = preprocess(inputs)  # Move to the cuda.
        num_samples += len(inputs[0])  # Update the number of samples.
        samples = model.module.inputs_to_struct(inputs)  # Make it struct.
        model.eval() #
        outs = model(inputs)  # Compute the output.
        outs = get_model_outs(model, outs)  # From output to struct
        (preds, task_accuracy_batch) = multi_label_accuracy_base(outs, samples)  # Compute the accuracy on the batch
        num_correct_pred += task_accuracy_batch.sum()  # Sum all accuracies on the batches.
    return num_correct_pred / num_samples  # Compute the mean.

'''
def multi_label_accuracy(outs, samples, nclasses):
    preds, task_accuracy = multi_label_accuracy_base(outs, samples)
    task_accuracy = task_accuracy.mean(axis=1)
    return preds, task_accuracy

def multi_label_accuracy_weighted_loss(outs, samples):
    preds, task_accuracy = multi_label_accuracy_base(outs, samples)
    return p
'''