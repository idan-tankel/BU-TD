import copy
import os
import torch
from training.Utils import tuple_direction_to_index
import argparse
import torch.nn as nn


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
    model.load_state_dict(checkpoint['model_state_dict'])  # Loading the saved weights.
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
    direction_dir, _ = tuple_direction_to_index(parser.num_x_axis, parser.num_y_axis, direction_tuple,
                                                parser.ndirections,
                                                task_id)
    task_id = torch.tensor(task_id)
    # The new task vector.
    New_task_flag = torch.nn.functional.one_hot(task_id, parser.ntasks)
    # The new direction vector.
    New_direction_flag = torch.nn.functional.one_hot(direction_dir, parser.ndirections)
    # Concat into one flag.
    New_flag = torch.concat([New_direction_flag, New_task_flag], dim=0).float()
    # Expand into one flag.
    return New_flag.unsqueeze(dim=0)


def set_model(model, state_dict):
    """
    Set model state by state dict.
    """
    model.load_state_dict(copy.deepcopy(state_dict))
