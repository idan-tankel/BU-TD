import argparse
from types import SimpleNamespace

import torch
from torch import device
import torch.nn as nn
from torch.utils.data import DataLoader
from supp.general_functions import preprocess
from supp import Dataset_and_model_type_specification as DMS
from typing import Union

CE = nn.CrossEntropyLoss(reduction='none')
dev = device("cuda") if torch.cuda.is_available() else device("cpu")

# Accuracy


def multi_label_accuracy_base(outs: Union[SimpleNamespace, object], samples: object, compound_all=False,occurence_only=True) -> tuple:
    """
    The base class for all modes.
    Here for each head we compute its accuracy according to the model out and label task.
    Args:
        outs: The model outs.
        samples: The samples.

    Returns: The predictions, the task accuracy.

    """
    total_number_of_tasks = 4
    cur_batch_size = samples.image.shape[0]
    predictions = torch.argmax(input=outs.task, dim=1, keepdim=False)
    # TODO: update the structure of the data_loader to include the flag
    if occurence_only:
        direction_one_hot = samples.label_existence
        predictions_by_correct_task = torch.mul(
            predictions, direction_one_hot)
        labels_by_correct_task = torch.mul(samples.label_task,direction_one_hot)
        # since the border class is not zero,naturally it will not be added to the number of successes - it will be 0 - 0 = 0
        number_of_errors = (predictions_by_correct_task - labels_by_correct_task).count_nonzero()
        total_number_of_tasks = samples.label_existence.count_nonzero()
        task_accuracy = (total_number_of_tasks - number_of_errors) /total_number_of_tasks
        assert task_accuracy <= 1
        return predictions, task_accuracy
        


    direction_one_hot = samples.flag[:,
                                     0:total_number_of_tasks].type(torch.int64)
    if not compound_all:
        direction_map = direction_one_hot.argmax(dim=1)
        # choose for each sample the col (direction) appropriate for the sample
        predictions_by_correct_task = predictions.gather(
            dim=1, index=direction_map.unsqueeze(1)).squeeze(1)
        label_task = samples.label_task.squeeze(-1)
        # Take the single original label for each sample
        task_accuracy = (
            (predictions_by_correct_task == label_task).float()).sum() / cur_batch_size
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


def multi_label_accuracy(outs: object, samples: object):
    """
    .. note:: Deprecated in 11_0
    ### Marked for deprecation ###
    return the task accuracy mean over all samples.

    Args:
        outs: The model outs.
        samples: The samples.

    Returns: The predictions and task accuracy over the batch.

    """
    preds, task_accuracy = multi_label_accuracy_base(outs, samples)
    # per single example
    avg_task_accuracy = task_accuracy.mean()
    return preds, avg_task_accuracy
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
    task_accuracy = task_accuracy.sum() / loss_weight.sum()  # per single example
    return preds, task_accuracy
# Loss


def multi_label_loss_base(outs: object, samples: object, guided:bool=False):
    """
    Here for each head we compute its loss according to the model out and label task.
    Args:
        outs: The model outs.
        samples: The samples.

    Returns: The loss over all heads.

    """
    loss_tasks = torch.zeros(samples.label_task.shape)
    direction_one_hot = samples.flag[:, 0:4].type(torch.int64)
    # since the directions are encoded as one hot vectors
    # we can use simple BMM to get the output by the correct direction and zero out all
    # without using more complex functions as gather...

    # direction_map = direction_one_hot.argmax(dim=1).view(-1,1,1)
    # use gather and scatter of torch to get the loss of each task
    # task_output = [outs.task[k,:,directions_flags[k]] for k in range(samples.flag.shape[0])]
    # task_output = outs.task.gather(dim=2,index=direction_map.repeat(1,48,1))
    label_task = samples.label_task.squeeze(dim=1).type(torch.LongTensor).to(dev)
    if guided:
        task_output = torch.bmm(
            outs.task, direction_one_hot.unsqueeze(2).type(torch.float))
    else:
        # 2 is the number of tasks, representing 47 different tasks - one for each char
        task_output = outs.task
        label_task *= samples.label_existence.type(torch.LongTensor).to(dev)
        task_output *= samples.label_existence.unsqueeze(2).to(dev)
    task_output = task_output.squeeze(dim=1)
    # TODO convert this part to scatter_add_
    # in order to verify that the CE will taken according to the classes that do appear in the image only
    # out of 48 available classes we have to multiply the CE by the existence of the class in the image (one hot)
    # compute the loss
    loss_tasks = CE(task_output, label_task)
    loss_tasks_old = torch.zeros_like(loss_tasks).to(dev,non_blocking=True)
    for k in range(48):
        # task_output = outs[:, :, k]  # For each task extract its last layer (shape 10,48)
        # label_task = samples.label_task[:, k]  # The label for the loss
        taskk_out = task_output[:,:,k]
        label_taskk = label_task[:,k]
        loss_tasks_old[:, k] += CE(input=taskk_out,target=label_taskk)  # Assign for each task its loss. (shape )
    assert loss_tasks_old == loss_tasks
    # loss_tasks = CE(outs.task, samples.label_task)
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
    *** MARKED FOR DEPRECATION ***
    The loss over all existing characters in the batch.
    Args:
        outs: The model outputs.
        samples: The samples.

    Returns: The weighted loss over all existing characters, for not existing the loss is zero.

    """
    losses_task = multi_label_loss_base(outs, samples)
    loss_weight = samples.label_existence
    losses_task = losses_task * loss_weight
    # a single valued result for the whole batch
    loss_task = losses_task.sum() / loss_weight.sum()
    return loss_task


def UnifiedCriterion(opts: argparse.ArgumentParser, inputs: list[torch.Tensor], outs: list[torch.Tensor], model=None) -> torch.Tensor:
    """Loss function based on BCE loss

    Args:
        opts (argparse): The model options
        inputs (list[torch.Tensor]): _description_
        outs (list[torch.Tensor]): _description_

    Returns:
        _type_: _description_
    """
    if model is None:
        model = opts.model
    outs = model.outs_to_struct(outs)  # The output from all the streams.
    samples = DMS.inputs_to_struct(inputs)  # Make samples from the raw data.
    if not isinstance(opts, argparse.ArgumentParser):
        opts = opts.Losses
    loss = 0.0  # The general loss.
    if opts.use_bu1_loss:
        try:
            loss_occ = opts.bu1_loss(outs.occurence,
                                     samples.label_existence)  # compute the binary existence classification loss
        except KeyError:
            raise KeyError(
                "The model does not have an occurence stream, please check the model architecture and the losses flags")
        loss += loss_occ  # Add the occurrence loss.

    if opts.use_bu2_loss:
        loss_task = opts.bu2_loss(outs, samples)  # Compute the BU2 loss.
        # TODO change the opts.bu2_loss to not be under the opts object
        loss += loss_task

    return loss


def accuracy(parser: nn.Module, test_data_loader: DataLoader) -> float:
    """
    *** Deprecated ***
    Since that function accumulates the accuracy over the whole dataset, it is not suitable for large datasets. We will use the `wandb.log` function instead.
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
        (_, task_accuracy_batch) = multi_label_accuracy_weighted(
            outs, samples)  # Compute the accuracy on the batch
        # Sum all accuracies on the batches.
        num_correct_pred += task_accuracy_batch.sum()
    return num_correct_pred / num_samples  # Compute the mean.
