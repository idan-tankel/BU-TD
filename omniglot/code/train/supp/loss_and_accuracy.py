import torch.nn as nn
from supp.measurments import *
import torch
from supp.data_functions import *
import argparse
from supp.Data_and_structs import *

# from utils.training_functions import test_step\
# Accuracy#
# TODO-change to support the NOFLAG mode.
def multi_label_accuracy_base(outs: object, samples: object, stage:int, num_outputs: int = 1) -> tuple:
    """
    :param outs:Outputs from all the model, including BU1,TD,BU2 outputs.
    :param samples: The samples we train in the current step.
    :param num_outputs: The number outputs the model should return(usually 1 or the number of characters in the image).
    :return: The predictions and the accuracy on the batch.
    """
    cur_batch_size = samples.image.shape[0]
    predictions = torch.zeros((cur_batch_size, num_outputs), dtype=torch.int).to(dev, non_blocking=True)
    task_output = outs.task  # For each task extract its predictions.
    task_pred = torch.argmax(task_output, axis=1).squeeze()  # Find the highest probability in the distribution
    predictions = task_pred  # assign for each task its predictions
    label_task = samples.label_task
    if stage == 2:
     task_accuracy = (predictions == label_task).float().sum(dim = 0) / (num_outputs)   # Compare the number of matches and normalize by the batch size*num_outputs.
    else:
     task_accuracy = (predictions == label_task).float().sum(dim=0) == 2 / (num_outputs)
    return predictions, task_accuracy  # return the predictions and the accuracy.
 #  task_accuracy = (predictions == label_task).float().sum(dim = 1) == 2 / (num_outputs) 

# Loss#
# TODO-change to support the NOFLAG MODE.
# loss_task_multi_label = nn.CrossEntropyLoss(reduction='none').to(dev)
class multi_label_loss_base:
    def __init__(self, parser=None, num_outputs=1):
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
        task_output = outs.task.squeeze()  # For each task extract its last layer.
        label_task = samples.label_task  # The label for the loss
        loss_task = self.classification_loss(task_output, label_task)  # compute the loss
        return loss_task  # return the task loss


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
        self.use_td_loss = opts.use_td_loss
        self.use_bu2_loss = opts.use_bu2_loss
        self.bu1_loss = opts.bu1_loss
        self.td_loss = opts.td_loss
        self.bu2_loss = opts.bu2_loss
        self.inputs_to_struct = cyclic_inputs_to_strcut
        self.opts = opts
 # outs_to_struct
# TODO - CHANGE TO NOT PROVIDE MODEL.
    def __call__(self, samples: list[torch], outs: list) -> float:
        """
        :param inputs: Input to the model.
        :param outs: Output from the model.
        :return: The combined loss over all the stream.
        """
        # from the raw data.
        loss = 0.0  # The general loss.
        if self.use_bu1_loss:
            loss_occ = self.bu1_loss(outs.occurrence_out, samples.label_existence)  # compute the binary existence classification loss
            loss += loss_occ  # Add the occurrence loss.
        # TODO - DELETE this option.
        if self.use_td_loss:
            loss_seg_td = self.td_loss(outs.td_head, samples.seg)  # compute the TD segmentation loss.
            loss_bu1_after_convergence = 1
            loss_td_after_convergence = 100
            ratio = loss_bu1_after_convergence / loss_td_after_convergence
            loss += ratio * loss_seg_td  # Add the TD segmentation loss.
        if self.use_bu2_loss:
            loss_task = self.bu2_loss(outs, samples)  # Compute the BU2 loss.
            loss += loss_task
        return loss

class CYCLICUnifiedLossFun:
    def __init__(self,opts):
        self.base_loss = UnifiedLossFun(opts)
        self.opts = opts

    def __call__(self, inputs: list[torch], outs_list: list) -> float:
        loss = 0.0
        stages = self.opts.stages
        # Make samples
        outs = get_model_outs(self.opts, outs_list)
        if 0 in stages:
          samples = cyclic_inputs_to_strcut(inputs, stage = 0)
          loss_stage_1 = self.base_loss(self.opts.model,samples,outs[0])
          loss += loss_stage_1
        if 1 in stages:
          samples = cyclic_inputs_to_strcut(inputs, stage=1)
          loss_stage_2 = self.base_loss(self.opts.model, samples, outs[1])
          loss+=loss_stage_2
        if 2 in stages:
          samples = cyclic_inputs_to_strcut(inputs, stage=2)
          loss_stage_3 = self.base_loss(self.opts.model, samples, outs[2])
          loss += loss_stage_3
        return loss

def accuracy(opts: nn.Module, test_data_loader: DataLoader) -> float:
    """
    :param opts:The model options to compute its accuracy.
    :param test_data_loader: The data.
    :return:The computed accuracy.
    """
    num_correct_pred, num_samples = (0.0, 0.0)

    for inputs in test_data_loader:  # Running over all inputs
        inputs = preprocess(inputs)  # Move to the cuda.
        num_samples += len(inputs[0])  # Update the number of samples.
        samples = cyclic_inputs_to_strcut(inputs, stage = 2)  # Make it struct.
        opts.model.eval() #
        outs = opts.model(inputs)  # Compute the output.
        outs = get_model_outs(opts, outs)  # From output to struct
        outs = outs[2]
        (_, task_accuracy_batch) = multi_label_accuracy_base(outs, samples,stage=2)  # Compute the accuracy on the batch
        num_correct_pred += task_accuracy_batch.sum()  # Sum all accuracies on the batches.
    return num_correct_pred / num_samples  # Compute the mean.


'''
def multi_label_accuracy(outs, samples, nclasses):
    preds, task_accuracy = multi_label_accuracy_base(outs, samples)
    task_accuracy = task_accuracy.mean(axis=1)
    return preds, task_accuracy

def multi_label_accuracy_weighted_loss(outs, samples):
    preds, task_accuracy = multi_label_accuracy_base(outs, samples)
    return preds, task_accuracy
'''
