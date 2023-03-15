"""
Here we define the function utils.
"""
import argparse
import os
from typing import Iterator, Optional

import torch
import torch.optim as optim
from torch import Tensor
import torch.nn as nn
from Data_Creation.src.Create_dataset_classes import Sample


def folder_size(path: str) -> int:
    """
    Returns the number of files in a given folder.
    Args:
        path: Path to a language file.

    Returns: Number of files in the folder
    """
    return len(list(os.scandir(path)))


def create_dict(path: str, offset: int = 0) -> dict:
    """
    Creates a dictionary assigning for each path in the folder the number of files in it.
    Args:
        path: Path to all raw Omniglot languages.
        offset: The offset for the 'initial tasks' place.

    Returns: Dictionary of number of characters per language

    """
    dict_language = {}
    for cnt, ele in enumerate(os.scandir(path)):  # for language in Omniglot_raw find the number of characters in it.
        dict_language[cnt + offset] = folder_size(ele)  # Find number of characters in the folder.

    return dict_language


def get_omniglot_dictionary(num_tasks: int, raw_data_folderpath: str) -> dict:
    """
    Getting the omniglot dictionary, for each task the number of characters in it.
    Args:
        num_tasks: The initial tasks set.
        raw_data_folderpath: The path to the raw data.

    Returns: A dictionary assigning for each task its number of characters.

    """
    nclasses = create_dict(path=raw_data_folderpath,
                           offset=1)  # Receiving for each task the number of characters in it.
    # nclasses[0] = sum( nclasses[51-task-2] for task in range(num_tasks))  # receiving
    # number of characters in the initial tasks.
    nclasses = {k: v for k, v in sorted(nclasses.items(), key=lambda item: item[1])}
    nclasses_New = {}
    for i, key in enumerate(nclasses.keys()):
        nclasses_New[i] = nclasses[key]
    nclasses_New[50] = sum(nclasses_New[50 - task - 2] for task in range(num_tasks))
    return nclasses_New


def get_laterals(laterals: list[Tensor], layer_id: int, block_id: int = 0) -> Optional[Tensor]:
    """
    Returns the lateral connections associated with the block in the layer.
    Args:
        laterals: All lateral connections from the previous stream, if exists.
        layer_id: The layer id in the stream.
        block_id: The block id in the layer.

    Returns: All the lateral connections associate with the block(usually 3).

    """
    try:
        # Trying to access that index.
        return laterals[layer_id][block_id]
    except (IndexError, TypeError):
        return None


def num_params(params: Iterator) -> int:
    """
    Computing the number of parameters in a given list.
    Args:
        params: The list of parameters.

    Returns: The number of learnable parameters in the list.

    """
    num_param = 0
    for param in params:
        # For each parameter in the model we multiply all its shape dimensions.
        shape = torch.tensor(param.shape)  # Make a tensor.
        num_param += torch.prod(shape)  # Add to the sum.
    return num_param


def create_optimizer_and_scheduler(opts: argparse, learned_params: list, nbatches: int) -> \
        tuple[optim, optim.lr_scheduler]:
    """
    Create optimizer and a scheduler according to opts.
    Args:
        opts: The model options.
        learned_params: The learned parameters.
        nbatches: The number of batches in the data-loader

    Returns: Optimizer, scheduler.

    """

    optimizer = optim.Adam(params=learned_params, lr=opts.initial_lr, weight_decay=opts.wd )

    if opts.scheduler_type is optim.lr_scheduler.MultiStepLR:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                   milestones=[15], gamma=opts.gamma, last_epoch=-1)
    elif opts.scheduler_type is optim.lr_scheduler.CosineAnnealingLR:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    elif opts.scheduler_type is optim.lr_scheduler.StepLR:
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.2)
    elif opts.scheduler_type is optim.lr_scheduler.PolynomialLR:
        scheduler = optim.lr_scheduler.PolynomialLR(optimizer=optimizer, total_iters=10, power=1.0)
    else:
        scheduler = None

    return optimizer, scheduler


def preprocess(inputs: list[Tensor], device: str) -> list[Tensor]:
    """
    Move list of tensors to the device.
    Args:
        inputs: The list of inputs we desire to move to the device.
        device: The device we desire to transforms to.

    Returns: The same inputs but in another device.

    """
    inputs = [torch.tensor(inp) for inp in inputs]
    inputs = [inp.to(device) for inp in inputs]  # Moves the tensor into the device,
    # usually to the cuda.
    return inputs


def tuple_direction_to_index(num_x_axis: int, num_y_axis: int, direction: tuple, ndirections: int, language_idx: int = 0) \
        -> tuple[Tensor, Tensor]:
    """
    Compute given task tuple and task index the task index and
    task index. Args: num_x_axis: The neighbor radios we want to generalize to in the x-axis.
    num_y_axis: The neighbor radios we want to generalize to in the y-axis. direction: The task tuple.
    ndirections: The number of directions. language_idx: The task index.

    Returns: The direction index and the task index.

    """
    direction_idx = tuple_to_direction(num_y_axis=num_y_axis, num_x_axis=num_x_axis, direction=direction)
    index_dir = direction_idx + ndirections * language_idx
    return direction_idx, index_dir


def tuple_to_direction(num_x_axis: int, num_y_axis: int, direction: tuple) \
        -> Tensor:
    """
    Tuple of direction to single direction.
    Args:
        num_x_axis: The number of x-axis query.
        num_y_axis: The number of y-axis query.
        direction: The direction.

    Returns: The single direction id.

    """
    direction_x, direction_y = direction
    index_dir = (num_x_axis + 1) * (direction_x + num_x_axis) + (direction_y + num_y_axis)
    return index_dir


def struct_to_input(sample: Sample) -> tuple[Tensor, Tensor, Tensor]:
    """
    Returning the sample attributed.
    Args:
        sample: Sample to return its attributes.

    Returns: The sample's attributes: label_existence, label_all ,flag, query coordinate.

    """
    # The label existence, telling for each entry whether the class exists or not.
    label_existence = sample.label_existence
    # All characters arranged.
    label_all = sample.label_ordered
    # The coordinate we query about.
    query_coord = sample.query_coord
    return label_existence, label_all, query_coord


def load_model(model, results_dir, model_path: str) -> dict:
    """
    Loads and returns the model checkpoint as a dictionary.
    Args:
        model: The model we want to load into.
        model_path: The path to the model.
        results_dir: Trained model dir.

    Returns: The loaded checkpoint.

    """
    model_path = os.path.join(results_dir, model_path)  # The path to the model.
    checkpoint = torch.load(model_path)  # Loading the saved data.
    model.load_state_dict(checkpoint['model_state_dict'])  # Loading the saved weights.
    return checkpoint


def Expand(mod: Tensor, shapes: list) -> Tensor:
    """
    Expand the tensor in interleaved manner to match the neuron's shape.
    Args:
        mod: The modulations.
        shapes: The shape to multiply each dimension.

    Returns: The expanded modulations.

    """
    for dim, shape in enumerate(shapes):
        mod = torch.repeat_interleave(mod, shape, dim=dim)
    return mod


def load_pretrained_model(model: nn.Module, model_state_dict: dict) -> dict:
    """
    Load pretrained model without weight modulation.
    Args:
        model: The model.
        model_state_dict: The trained model state dict.

    Returns:

    """
    checkpoint = dict()
    for key, val in model.state_dict().items():
        if 'modulation' in key or 'mask' in key:
            checkpoint[key] = val
        else:
            checkpoint[key] = model_state_dict[key]
    return checkpoint
