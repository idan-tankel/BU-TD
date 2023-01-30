"""
Here we define the loss function, including multi label and weighted loss for baselines.
"""
import argparse

import torch.nn as nn
from ..Data.Structs import inputs_to_struct, outs_to_struct

CE = nn.CrossEntropyLoss(reduction='none')


# Here we define our loss functions.

def multi_label_loss(samples: inputs_to_struct, outs: outs_to_struct):
    """
    The average loss over all images in the batch.
    Args:
        samples: The inputs.
        outs: The model outputs.


    Returns: The mean loss over all inputs in the batch.

    """
    samples.label_task = samples.label_task.squeeze()
    losses_task = CE(input=outs.classifier, target=samples.label_task)  # Compute the CE loss without reduction.
    loss_task = losses_task.mean()  # Mean over all batch.
    return loss_task


def multi_label_loss_weighted(samples: inputs_to_struct, outs: outs_to_struct):
    """
    The loss over all existing characters in the batch(non-zero loss weight).
    Args:
        samples: The inputs.
        outs: The model outputs.

    Returns: The weighted loss over all existing characters, for not existing the loss is zero.

    """
    losses_task = CE(input=outs.classifier, target=samples.label_task)  # Compute the CE loss without reduction.
    loss_weight = samples.label_existence  # The loss weight for only existing characters.
    loss_weight = loss_weight.view(losses_task.shape)
    losses_task *= loss_weight  # Compute the loss only in the existing characters.
    loss_task = losses_task.sum() / loss_weight.sum()  # Mean the loss over all batch and characters.
    return loss_task


def UnifiedCriterion(opts: argparse, samples: inputs_to_struct, outs: outs_to_struct):
    """
    BU1 loss plus BU2 loss.
    Args:
        opts: The model options.
        samples:  The input inputs to the model.
        outs: The model out.

    Returns: The overall loss, including BU1 loss(if needed) and BU2.

    """
    loss = 0.0  # The general loss.
    if opts.use_bu1_loss:  # If we use BU1 loss.
        loss_occ = opts.bu1_loss(input=outs.occurrence_out,
                                 target=samples.label_existence)  # compute the binary existence classification loss
        loss += loss_occ  # Add the occurrence loss.

    loss_task = opts.bu2_loss(samples=samples, outs=outs)  # Compute the BU2 loss.
    loss += loss_task
    return loss
