import torch.nn as nn
from supp.measurments import *
import torch
from supp.data_functions import *
import argparse
from supp.Data_and_structs import *

# Accuracy#
# TODO-change to support the NOFLAG mode.
def multi_label_accuracy_base(outs: object, samples: object, stage:int, num_outputs: int = 1) -> tuple:
    """
    Args:
        outs: Outputs from all the model, including BU1,TD,BU2 outputs.
        samples: The samples we train/test in the current step.
        stage: The stage we compute its accuracy.
        num_outputs: The number outputs the model should return(usually 1 or the number of characters in the image).

    Returns: The predictions and the accuracy on the batch.

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
     task_accuracy = (predictions == label_task).float().sum(dim = 1) == 2 / (num_outputs) # To match a point we have to be right two times.
    return predictions, task_accuracy  # return the predictions and the accuracy.

# Loss#
class multi_label_loss_base:
    def __init__(self, num_outputs=1):
        """
        Args:
            num_outputs: The number of outputs.
        """
        self.num_outputs = num_outputs
        self.classification_loss = nn.CrossEntropyLoss(reduction='none').to(dev)

    def __call__(self, outs: object, samples: object) -> torch:
        """
        Args:
            outs: Outputs from all the model, including BU1,TD,BU2 outputs.
            samples: he samples we train/test in the current step.

        Returns: The loss on the batch.
        """
        task_output = outs.task.squeeze()  # For each task extract its last layer.
        label_task = samples.label_task  # The label for the loss
        loss_task = self.classification_loss(task_output, label_task)  # compute the loss
        return loss_task  # return the task loss

def multi_label_loss(outs: object, samples: object) -> float:
    """
    Args:
        outs: Outputs from all the model, including BU1,TD,BU2 outputs.
        samples: The samples we train/test in the current step.

    Returns: The average loss on the batch.
    """
    losses_task = multi_label_loss_base()(outs, samples)  # list of the losses per sample.
    return losses_task.mean()  # The average loss


class UnifiedLossFun:
    def __init__(self, opts: argparse) -> None:
      """
      Args:
          opts: The model options.
      """
      self.use_bu1_loss = opts.use_bu1_loss
      self.use_td_loss = opts.use_td_loss
      self.use_bu2_loss = opts.use_bu2_loss
      self.bu1_loss = opts.bu1_loss
      self.td_loss = opts.td_loss
      self.bu2_loss = opts.bu2_loss
      self.inputs_to_struct = cyclic_inputs_to_strcut
      self.opts = opts

    def __call__(self, samples: list[torch], outs: list) -> float:
        """
        Args:
            samples: The samples we train/test in the current stage.
            outs: The output from the model.

        Returns: The combined loss over all the streams.
        """
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
    def __init__(self,opts:argparse):
        """
        Args:
            opts: The model options.
        """
        self.base_loss = UnifiedLossFun(opts)
        self.opts = opts

    def __call__(self, inputs: list[torch], outs_list: list) -> float:
        """
        Args:
            inputs: The inputs for all stages.
            outs_list: The output from all stages.

        Returns: The loss from all stages.

        """
        loss = 0.0
        stages = self.opts.stages
        outs = get_model_outs(self.opts, outs_list)
        if 0 in stages:
          # Make samples
          samples = cyclic_inputs_to_strcut(inputs, stage = 0)
          loss_stage_1 = self.base_loss(samples,outs[0])
          loss += loss_stage_1

        if 1 in stages:
          # Make samples
          samples = cyclic_inputs_to_strcut(inputs, stage=1)
          loss_stage_2 = self.base_loss(samples, outs[1])
          loss+=loss_stage_2

        if 2 in stages:
          # Make samples
          samples = cyclic_inputs_to_strcut(inputs, stage=2)
          loss_stage_3 = self.base_loss(samples, outs[2])
          loss += loss_stage_3

        return loss

def accuracy(opts: nn.Module, test_data_loader: DataLoader,stage:int) -> float:
    """
    Args:
        opts: The model options to compute its accuracy.
        test_data_loader: The data we test.
        stage: The stage we test.

    Returns: The computed accuracy.
    """
    assert stage in [0, 1, 2]
    num_correct_pred, num_samples = (0.0, 0.0)
    for inputs in test_data_loader:  # Running over all inputs
        inputs = preprocess(inputs)  # Move to the cuda.
        num_samples += len(inputs[0])  # Update the number of samples.
        samples = cyclic_inputs_to_strcut(inputs, stage = stage)  # Make it struct.
        opts.model.eval() #
        outs = opts.model(inputs)  # Compute the output.
        outs = get_model_outs(opts, outs)  # From output to struct
        outs = outs[stage]
        (_, task_accuracy_batch) = multi_label_accuracy_base(outs, samples,stage = stage)  # Compute the accuracy on the batch
        num_correct_pred += task_accuracy_batch.sum()  # Sum all accuracies on the batches.
    return num_correct_pred / num_samples  # Compute the mean.

