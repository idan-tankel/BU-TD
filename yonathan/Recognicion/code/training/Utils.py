import argparse
import os
from typing import Union, Iterator

import torch
import torch.optim as optim


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
    nclasses = create_dict(raw_data_folderpath, offset=1)  # Receiving for each task the number of characters in it.
    #  nclasses[0] = sum(
    #      nclasses[51-task-2] for task in range(num_tasks))  # receiving number of characters in the initial tasks.
    nclasses = {k: v for k, v in sorted(nclasses.items(), key=lambda item: item[1])}
    nclasses_New = {}
    for i, key in enumerate(nclasses.keys()):
        nclasses_New[i] = nclasses[key]
    nclasses_New[50] = sum(nclasses_New[50 - task - 2] for task in range(num_tasks))
    return nclasses_New


def flag_to_idx(flag: torch) -> int:
    """
    From Flag get the id in which the flag is non-zero.
    Args:
        flag: The One hot flag.

    Returns: The id in which the flag is non-zero.

    """
    task = torch.argmax(flag, dim=1)[0]  # Finds the non-zero entry in the one-hot vector
    return task


def get_laterals(laterals: list[torch], layer_id: int, block_id: int) -> torch:
    """
    Returns the lateral connections associated with the block in the layer.
    Args:
        laterals: All lateral connections from the previous stream, if exists.
        layer_id: The layer id in the stream.
        block_id: The block id in the layer.

    Returns: All the lateral connections associate with the block(usually 3).

    """
    if laterals is None:  # If BU1, there are no lateral connections.
        return None
    try:
        # Trying to access that index.
        layer_lats = laterals[layer_id][block_id]
    except IndexError:
        layer_lats = None
    return layer_lats


def num_params(params: Union[Iterator]) -> int:
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


def create_optimizer_and_scheduler(opts: argparse, learned_params: list, nbatches_train: int) -> tuple:
    """
    Create optimizer and a scheduler according to opts.
    Args:
        opts: The model options.
        learned_params: The learned parameters.
        nbatches_train: The number of batches in the data-loader

    Returns: Optimizer, scheduler.

    """

    if opts.SGD:
        optimizer = optim.SGD(learned_params, lr=opts.initial_lr, momentum=opts.momentum, weight_decay=opts.wd)
    else:
        optimizer = optim.Adam(learned_params, lr=opts.base_lr, weight_decay=opts.wd)

    if opts.cycle_lr:
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=opts.base_lr, max_lr=opts.max_lr,
                                                step_size_up=nbatches_train // 2,
                                                cycle_momentum=False)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=nbatches_train // 4, gamma=0.9)

    return optimizer, scheduler


def preprocess(inputs: list[torch], device: str) -> list[torch]:
    """
    Move list of tensors to the device.
    Args:
        inputs: The list of inputs we desire to move to the device.
        device: The device we desire to transform to.

    Returns: The same inputs but in another device.

    """
    inputs = [inp.to(device) for inp in inputs]  # Moves the tensor into the device, usually to the cuda.
    return inputs


def tuple_direction_to_index(num_x_axis: int, num_y_axis: int, direction: tuple, ndirections: int, task_id: int) -> \
        tuple[torch, torch]:
    """
    Compute given direction tuple and task index the direction index and task index.
    Args:
        num_x_axis: The neighbor radios we want to generalize to in the x-axis.
        num_y_axis: The neighbor radios we want to generalize to in the y-axis.
        direction: The direction tuple.
        ndirections: The number of directions.
        task_id: The task index.

    Returns: The direction index and the task index.

    """
    direction_x, direction_y = direction
    index_dir = (num_x_axis + 1) * (direction_x + num_x_axis) + (direction_y + num_y_axis)
    direction_dir = torch.tensor(index_dir)
    index_dir = torch.tensor(index_dir + ndirections * task_id)
    return direction_dir, index_dir


def Compose_Flag(opts: argparse, flag: torch) -> tuple[torch, torch, torch]:
    """
    Compose the flag into three flags.
    Args:
        opts: The model opts.
        flag: The flag we desire to compose.

    Returns: The direction,task, arg flags.

    """
    direction_flag = flag[:, :opts.ndirections]  # The direction vector.
    task_flag = flag[:, opts.ndirections:opts.ndirections + opts.ntasks]  # The task vector.
    arg_flag = flag[:, opts.ndirections + opts.ntasks:]  # The argument vector.
    return direction_flag, task_flag, arg_flag


def Flag_to_task(opts: argparse, flag: torch) -> int:
    """
    Composes the flag and returns the task id.
    Args:
        opts: The model opts.
        flag: The flag

    Returns: The task index.

    """
    direction_flag, task_flag, _ = Compose_Flag(opts=opts, flag=flag)
    direction_id = flag_to_idx(direction_flag)  # The direction id.
    task_id = flag_to_idx(task_flag)  # The task id.
    idx = direction_id + opts.ndirections * task_id  # The task.
    return idx
