import argparse
import copy
import os
from typing import Callable

import torch
import torch.nn as nn
from avalanche.models.utils import avalanche_forward
from avalanche.training.utils import zerolike_params_dict
from torch import Tensor
from torch.utils.data import DataLoader

from training.Utils import preprocess
from training.Utils import tuple_direction_to_index


def load_model(model: nn.Module, results_path: str, model_path: str) -> dict:
    """
    Loads and returns the model checkpoint as a dictionary.
    Args:
        model_path: The path to the model.
        results_path: The path to results dir.
        model: The path to the model.

    Returns: The loaded checkpoint.

    """
    model_path = os.path.join(results_path, model_path)  # The path to the model.
    checkpoint = torch.load(model_path)  # Loading the saved data.
    model.load_state_dict(state_dict=checkpoint['model_state_dict'])  # Loading the saved weights.
    return checkpoint


def construct_flag(parser: argparse, task_id: int, direction_tuple: tuple):
    """
    Args:
        parser: The model opts.
        task_id: The task id.
        direction_tuple: The direction id.

    Returns: The new tasks, with the new task and direction.

    """
    # From the direction tuple to single number.
    direction_dir, _ = tuple_direction_to_index(num_x_axis=parser.num_x_axis, num_y_axis=parser.num_y_axis,
                                                direction=direction_tuple,
                                                ndirections=parser.ndirections,
                                                task_id=task_id)
    task_id = torch.tensor(task_id)
    # The new task vector.
    New_task_flag = torch.nn.functional.one_hot(task_id, parser.ntasks)
    # The new direction vector.
    New_direction_flag = torch.nn.functional.one_hot(direction_dir, parser.ndirections)
    # Concat into one flag.
    New_flag = torch.concat([New_direction_flag, New_task_flag], dim=0).float()
    # Expand into one flag.
    return New_flag.unsqueeze(dim=0)


def set_model(model: nn.Module, state_dict: dict):
    """
    Set model state by state dict.
    """
    model.load_state_dict(state_dict=copy.deepcopy(state_dict))


def compute_fisher_information_matrix(parser: argparse, model: nn.Module, criterion: Callable, dataloader: DataLoader,
                                      device: str, train: bool = True, norm=2) -> dict:
    """
    Compute fisher importance matrix for each parameter.
    Args:
        parser: The model opts
        model: The model we compute its coefficients.
        criterion: The loss criterion.
        dataloader: The train data-loader.
        device: The device.
        train: Whether the model should be in train/eval mode.
        norm: The norm to multiply the gradients.

    Returns: The importance coefficients.

    """
    model = model.train() if train else model.eval()
    importances = zerolike_params_dict(model.feature_extractor)  # Make empty coefficients.

    for i, batch in enumerate(dataloader):  # Iterating over the dataloader.
        x = preprocess(batch, device)  # Omit the ids and move to the device.
        x = parser.inputs_to_struct(x)  # Make a struct.
        model.zero_grad()  # Reset grads.
        out = avalanche_forward(model, x, task_labels=None)  # Compute output.
        out = parser.outs_to_struct(out)  # Make a struct.
        loss = criterion(parser, x, out)  # Compute the loss.
        loss.backward()  # Compute grads.
        for (k1, p), (k2, imp) in zip(model.feature_extractor.named_parameters(),
                                      importances):  # Iterating over the feature weights.
            assert k1 == k2
            if p.grad is not None:
                # Adding the grad**2.
                imp += torch.abs(p.grad.data.clone()).pow(norm)

    # average over mini batch length
    for _, imp in importances:
        imp /= float(len(dataloader))
    # Make dictionary.
    importances = dict(importances)
    return importances


def compute_quadratic_loss(current_model: nn.Module, previous_model: nn.Module, importance: dict,
                           device: str) -> Tensor.float:
    """
    Compute quadratic loss.
    Very common in regularization methods to use such loss.
    Args:
        current_model: The current model.
        previous_model: The previous model.
        importance: The per-weight importance.
        device: The device the computation on.

    Returns: The quadratic loss.

    """
    penalty = torch.tensor(0).float().to(device)

    for (name, param), (same_name, param_old) in zip(current_model.feature_extractor.named_parameters(),
                                                     previous_model.feature_extractor.named_parameters()):
        penalty += torch.sum(importance[name] * (param - param_old).pow(2))
        # Update the new loss.
    return penalty
