import torch.nn as nn
# from measurments import *
import torch
import numpy as np
# from data_functions import *
import argparse
# import DataLoader
from torch.utils.data import DataLoader
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
from supplmentery.get_model_outs import get_model_outs
from v26.funcs import preprocess

# TODO note that there is very similar code in the `v26.accuracy_funcs.py` file and in `v26.functions.loses.py` file.



# from utils.training_functions import test_step\
# Accuracy#
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
    predictions = torch.argmax(input=outs.task, dim=1, keepdim=False)
    # for k in range(num_outputs):
    #     task_output = outs.task[:, :, k]  # For each task extract its predictions.
    #     task_pred = torch.argmax(task_output, axis=1)  # Find the highest probability in the distribution
    #     predictions[:, k] = task_pred  # assign for each task its predictions
    direction_one_hot = samples.flag[:,0:4].type(torch.int64)
    direction_map = direction_one_hot.argmax(dim=1)
    predictions_by_correct_task = predictions.gather(dim=1, index=direction_map.unsqueeze(1)).squeeze(1)
    label_task = samples.label_task.squeeze(1)
    # TODO change this to support multi head architecture.
    # instead of using the label_task against all the heads in 
    task_accuracy = ((predictions_by_correct_task == label_task).float() / (
        num_outputs)).sum()  # Compare the number of matches and normalize by the batch size*num_outputs.
    return predictions, task_accuracy  # return the predictions and the accuracy.


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
         # infer the task_id from the flag of each sample in the batch
         # chech whenever the flag is 1 or 0 for each example in the batch
        loss_tasks = torch.zeros(samples.label_task.shape).to(dev, non_blocking=False)
        direction_one_hot = samples.flag[:,0:4].type(torch.int64)
        direction_map = direction_one_hot.argmax(dim=1).view(-1,1,1)
        # use gather and scatter of torch to get the loss of each task
        # task_output = [outs.task[k,:,directions_flags[k]] for k in range(samples.flag.shape[0])]
        task_output = outs.task.gather(dim=2,index=direction_map.repeat(1,48,1))
        task_output = task_output.squeeze(dim=2)
        # TODO convert this part to scatter_add_
        label_task = samples.label_task.squeeze(dim=1).type(torch.LongTensor).to(dev)
        loss_task = self.classification_loss(task_output, label_task)  # compute the loss
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
    losses_task = multi_label_loss_base()(outs, samples)  # list of the losses per sample.
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
        samples = self.inputs_to_struct(inputs)  # Make samples from the raw data.
        loss = 0.0  # The general loss.
        if self.use_bu1_loss:
            loss_occ = self.bu1_loss(outs.occurence,
                                     samples.label_existence)  # compute the binary existence classification loss
            loss += loss_occ  # Add the occurrence loss.
        if self.use_td_loss:
            loss_seg_td = self.td_loss(outs.td_head, samples.seg)  # compute the TD segmentation loss.
            loss_bu1_after_convergence = 1
            loss_td_after_convergence = 100
            ratio = loss_bu1_after_convergence / loss_td_after_convergence
            loss += ratio * loss_seg_td  # Add the TD segmentation loss.
        if self.use_bu2_loss:
            loss_task = self.bu2_classification_loss(outs, samples)  # Compute the BU2 loss.
            loss += loss_task
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
        samples = opts.inputs_to_struct(inputs)  # Make it struct.
        
        outs = opts.model(inputs)  # Compute the output.
        outs = get_model_outs(opts.model, outs)  # From output to struct
        opts.model.eval()
        ( _ , task_accuracy_batch) = multi_label_accuracy_base(outs, samples)  # Compute the accuracy on the batch
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
