from v26.funcs import preprocess
from supplmentery.get_model_outs import get_model_outs
from doctest import Example
from functools import total_ordering
import torch.nn as nn
# from measurments import *
import torch
import numpy as np
# from data_functions import *
import argparse
# import DataLoader
from torch.utils.data import DataLoader
from multipledispatch import dispatch
dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

# TODO note that there is very similar code in the `v26.accuracy_funcs.py` file and in `v26.functions.loses.py` file.


# from utils.training_functions import test_step\
# Accuracy#
# TODO-change to support the NOFLAG mode.
def multi_label_accuracy_base(outs: object, samples: object, compound_all: bool = False) -> tuple:
    """
    :param outs:Outputs from all the model, including BU1,TD,BU2 outputs.
    :param samples: The samples we train in the current step.
    :param num_outputs: The number outputs the model should return(usually 1 or the number of characters in the image).
    :return: The predictions and the accuracy on the batch.


    NOTE - the outs and the samples are structured - after inputs_to_struct function and get_model_outs function.
    """
    total_number_of_tasks = 4
    cur_batch_size = samples.image.shape[0]
    predictions = torch.argmax(input=outs.task, dim=1, keepdim=False)
    direction_one_hot = samples.flag[:,
                                     0:total_number_of_tasks].type(torch.int64)
    if not compound_all:
        direction_map = direction_one_hot.argmax(dim=1)
        # choose for each sample the col (direction) appropriate for the sample
        predictions_by_correct_task = predictions.gather(
            dim=1, index=direction_map.unsqueeze(1)).squeeze(1)
        label_task = samples.label_task.squeeze(1)
        # Take the single original label for each sample
        task_accuracy = (
            (predictions_by_correct_task == label_task).float()).sum()
        # task_accuracy here is the number of correct predictions
        # Compare the number of matches and normalize by the batch size*num_outputs.
    else:
        # need to do a one hot multiplication to get the indicator in the direction map which is now (batch_size,number_of_directions)
        predictions_by_correct_task = torch.mul(
            predictions, direction_one_hot)  # which predictions to take
        # now, since we are comparing against label_all and not label_task, we should get to test all the directions together.
        # now we shell focus on a specific argument of samples.label_all_directions
        # to find this argument, we shell keep in some location the position of the requested argument (occurence task)
        labels_all_directions = torch.mul(
            samples.label_all_directions, direction_one_hot.unsqueeze(1))
        things_to_test = torch.zeros(
            cur_batch_size, total_number_of_tasks).to(dev)
        for number_of_sample, row, arg in zip(range(cur_batch_size), samples.label_all, samples.arg):
            row_of_arg, col_of_arg = (row == arg).nonzero(as_tuple=True)
            things_to_test[number_of_sample,
                           :] = labels_all_directions[number_of_sample, col_of_arg, :]

            # now, these are the row + col that have been found
            # Add then 1 as an indicator to the propriate example in the index_of_arguments_location
        # since if we don't take all, to create 0=0 in the next comparison
        number_of_errors = (
            things_to_test - predictions_by_correct_task).count_nonzero()
        task_accuracy = cur_batch_size*2 - number_of_errors
        # this counts also zero = zero
        # this depends on the assumption of things_to_test and predictions_by_correct_task are multiplied by the same one_hot filter
    # return the predictions and the accuracy.
    return predictions, task_accuracy


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
        self.classification_loss = nn.CrossEntropyLoss(
            reduction='none').to(dev)

    def __call__(self, outs: object, samples: object) -> torch:
        """
         :param outs:Outputs from all the model, including BU1,TD,BU2 outputs.
         :param samples: The samples we train in the current step.
         :param num_outputs: The number outputs the model should return(usually 1 or the number of characters in the image).
         :return:The loss on the batch.
         """
        # infer the task_id from the flag of each sample in the batch
        # chech whenever the flag is 1 or 0 for each example in the batch
        loss_tasks = torch.zeros(samples.label_task.shape).to(
            dev, non_blocking=False)
        direction_one_hot = samples.flag[:, 0:4].type(torch.int64)

        # since the directions are encoded as one hot vectors
        # we can use simple BMM to get the output by the correct direction and zero out all
        # without using more complex functions as gather...

        # direction_map = direction_one_hot.argmax(dim=1).view(-1,1,1)
        # use gather and scatter of torch to get the loss of each task
        # task_output = [outs.task[k,:,directions_flags[k]] for k in range(samples.flag.shape[0])]
        # task_output = outs.task.gather(dim=2,index=direction_map.repeat(1,48,1))

        task_output = torch.bmm(
            outs.task, direction_one_hot.unsqueeze(2).type(torch.float))
        task_output = task_output.squeeze(dim=2)
        # TODO convert this part to scatter_add_
        label_task = samples.label_task.squeeze(
            dim=1).type(torch.LongTensor).to(dev)
        loss_task = self.classification_loss(
            task_output, label_task)  # compute the loss
        # for k in range(self.num_outputs):
        #     # task_output = outs.task[:, :, k]  # For each task extract its last layer (shape 10,48)
        #     label_task = samples.label_task[:, k]  # The label for the loss
        #     loss_tasks[:, k] = loss_task  # Assign for each task its loss. (shape )
        return loss_task  # return the task loss


def multi_label_loss(outs: object, samples: object) -> float:
    """
    :param outs:Outputs from all the model, including BU1,TD,BU2 outputs.
    :param samples: The samples we train in the current step.
    :return: The average loss on the batch.
    """
    losses_task = multi_label_loss_base()(
        outs, samples)  # list of the losses per sample.
    return losses_task.mean()  # The average loss


class UnifiedLossFun:
    def __init__(self, loss_opts: argparse) -> None:
        """
     :param opts: tells which losses to use and the loss functions.
     """
        self.use_bu1_loss = loss_opts.use_bu1_loss
        self.use_td_loss = loss_opts.use_td_loss
        self.use_bu2_loss = loss_opts.use_bu2_loss
        self.bu1_loss = loss_opts.bu1_loss
        self.td_loss = loss_opts.td_loss
        self.bu2_classification_loss = loss_opts.bu2_loss
        self.inputs_to_struct = loss_opts.inputs_to_struct

    def __call__(self, model, inputs: list[torch], outs: list) -> float:
        """
        :param inputs: Input to the model.
        :param outs: Output from the model.
        :return: The combined loss over all the stream.
        """
        outs = get_model_outs(model, outs)  # The output from all the streams.
        # Make samples from the raw data.
        samples = self.inputs_to_struct(inputs)
        loss = 0.0  # The general loss.
        if self.use_bu1_loss:
            loss_occ = self.bu1_loss(outs.occurence,
                                     samples.label_existence)  # compute the binary existence classification loss
            loss += loss_occ  # Add the occurrence loss.
        if self.use_td_loss:
            # compute the TD segmentation loss.
            loss_seg_td = self.td_loss(outs.td_head, samples.seg)
            loss_bu1_after_convergence = 1
            loss_td_after_convergence = 100
            ratio = loss_bu1_after_convergence / loss_td_after_convergence
            loss += ratio * loss_seg_td  # Add the TD segmentation loss.
        if self.use_bu2_loss:
            # Compute the BU2 loss.
            loss_task = self.bu2_classification_loss(outs, samples)
            loss += loss_task
        return loss


@dispatch(object, DataLoader, nn.Module)
def accuracy(opts: object, test_data_loader: DataLoader, model: nn.Module) -> float:
    """
    :param opts:The model options to compute its accuracy.
    :param test_data_loader: The data.
    :return:The computed accuracy.
    """
    num_correct_pred, num_samples = (0.0, 0.0)

    for inputs in test_data_loader:  # Running over all inputs
        inputs = preprocess(inputs)  # Move to the cuda.
        num_samples += len(inputs[0])  # Update the number of samples.
        samples = opts.inputs_to_struct(inputs)  # Make it struct.

        outs = model(inputs)  # Compute the output.
        outs = get_model_outs(model, outs)  # From output to struct
        model.eval()
        (_, task_accuracy_batch) = multi_label_accuracy_base(
            outs, samples)  # Compute the accuracy on the batch
        # Sum all accuracies on the batches.
        num_correct_pred += task_accuracy_batch.sum()
    return num_correct_pred / num_samples  # Compute the mean.


# for compound instructions only
@dispatch(object, DataLoader, nn.Module, int)
def accuracy(opts: object, data_loader: DataLoader, model: nn.Module, ntasks: int) -> float:
    """
    :param opts:The model options to compute its accuracy.
    :param test_data_loader: The data.
    :return:The computed accuracy.
    """
    num_correct_pred, num_samples = (0.0, 0.0)

    for inputs in data_loader:  # Running over all inputs
        inputs = preprocess(inputs)  # Move to the cuda.
        num_samples += len(inputs[0])  # Update the number of samples.
        samples = opts.inputs_to_struct(inputs)  # Make it struct.
        samples.label_all
        border_value = 47
        # pad all directions with 47 which is the N/A class
        label_all_pad = torch.nn.functional.pad(
            samples.label_all, (1, 1, 1, 1), mode='constant', value=border_value)
        # 4 is for the 4 directions!!!!
        # this is kind of a magic number
        # 5 = 4+1 one to keep the original value instead of working by location
        kernel = torch.zeros(4, 1, 3, 3).type(
            dtype=torch.float).to(device=label_all_pad.device)
        kernel[0, 0, 1, 1]
        kernel[0, 0, 1, 2] = 1.0   # left
        kernel[1, 0, 1, 0] = 1.0   # right
        kernel[2, 0, 0, 1] = 1.0   # up
        kernel[3, 0, 2, 1] = 1.0   # down
        ntasks_all = 4
        # 4 is the number of tasks, and as the flag is splitted - the first 4 are the adj_type and the last 4 are the char
        samples.arg = samples.flag[:, ntasks_all:].argmax(dim=1)
        # TODO change this to use the samples class! all this thing should be in the __init__ of the class!!
        # we have added one group - in the original example batch_size = 1 and then things worked out
        samples.label_all_directions = torch.nn.functional.conv2d(label_all_pad.unsqueeze(
            1).type(torch.float), kernel).squeeze(dim=2).permute(0, 2, 1)
        # get location of the argument to go back to label_all_directions

        inputs[5][:, :ntasks] = torch.ones_like(
            inputs[5][:, :ntasks])  # flag modification to ones only
        # TODO change this this sould be only on inference!
        outs = model(inputs)  # Compute the output.
        outs = get_model_outs(model, outs)  # From output to struct
        model.eval()
        (_, task_accuracy_batch) = multi_label_accuracy_base(
            outs, samples, compound_all=True)  # Compute the accuracy on the batch
        # Sum all accuracies on the batches.
        num_correct_pred += task_accuracy_batch.sum()
    return num_correct_pred / (num_samples*ntasks)  # Compute the mean.
