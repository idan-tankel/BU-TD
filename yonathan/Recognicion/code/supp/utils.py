import argparse
import torch
import torch.optim as optim
import os


def folder_size(path: str) -> int:
    """
    Returns the number of files in a given folder.
    Args:
        path: Path to a language file.

    Returns: Number of files in the folder
    """
    return len([_ for _ in os.scandir(path)])


def create_dict(path: str) -> dict:
    """
    Creates a dictionary assigning for each path in the folder the number of files in it.
    Args:
        path: Path to all raw Omniglot languages.

    Returns: Dictionary of number of characters per language

    """
    dict_language = {}
    for cnt, ele in enumerate(os.scandir(path)):  # for language in Omniglot_raw find the number of characters in it.
        dict_language[cnt] = folder_size(ele)  # Find number of characters in the folder.
    return dict_language


def get_omniglot_dictionary(initial_tasks: list, raw_data_folderpath: str) -> list:
    """
    Getting the omniglot dictionary, for each task the number of characters in it.
    Args:
        initial_tasks: The initial tasks set.
        raw_data_folderpath: The path to the raw data.

    Returns: A dictionary assigning for each task its number of characters.

    """

    nclasses = create_dict(raw_data_folderpath, offset=1)  # Receiving for each task the number of characters in it.
    nclasses[0] = sum(nclasses[task] for task in initial_tasks)  # receiving number of characters in the initial tasks.
    return nclasses


def flag_to_task(flag: torch) -> int:
    """
    From Flag get the id in which the flag is non-zero.
    Args:
        flag: The One hot flag.

    Returns: The id in which the flag is non-zero.

    """
    task = torch.argmax(flag, dim=1)[0]  # Finds the non zero entry in the one-hot vector
    return task


def get_laterals(laterals: list[torch], layer_id: int, block_id: int) -> torch:
    """
    Returns the lateral connections associated with the layer, block.
    Args:
        laterals: All lateral connections from the previous stream, if exists.
        layer_id: The layer id in the stream.
        block_id: The block id in the layer.

    Returns: All the lateral connections associate with the block(usually 3).

    """
    if laterals is None:  # If BU1, there are not any lateral connections.
        return None
    if len(laterals) > layer_id:  # assert we access to an existing layer.
        layer_laterals = laterals[layer_id]  # Get all lateral associate with the layer.
        if type(layer_laterals) == list and len(
                layer_laterals) > block_id:  # If there are some several blocks in the layer we access according to block_id.
            return layer_laterals[block_id]  # We return all lateral associate with the block_id.
        else:
            return layer_laterals  # If there is only 1 lateral connection in the block we return it.
    else:
        return None


def num_params(params: list) -> int:
    """
    Computing the number of parameters in a given list.
    Args:
        params: The list of parameters.

    Returns: The number of learnable parameters in the list.

    """
    nparams = 0
    for param in params:  # For each parameter in the model we sum its parameters
        cnt = 1
        for p in param.shape:  # The number of params in each weight is the product if its shape.
            cnt = cnt * p
        nparams = nparams + cnt  # Sum for all params.
    return nparams


def create_optimizer_and_scheduler(opts: argparse, learned_params: list) -> tuple:
    """
    Create optimizer and a scheduler according to opts.
    Args:
        opts: The model options.
        learned_params: The learned parameters.

    Returns: Optimizer, scheduler.

    """

    if opts.SGD:
        optimizer = optim.SGD(learned_params, lr=opts.initial_lr, momentum=opts.momentum, weight_decay=opts.wd)
        if opts.cycle_lr:
            scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=opts.base_lr, max_lr=opts.max_lr,
                                                    step_size_up=opts.nbatches_train // 2, step_size_down=None,
                                                    mode='triangular', gamma=1.0, scale_fn=None,
                                                    cycle_momentum=True,
                                                    )
    else:
        optimizer = optim.Adam(learned_params, lr=opts.base_lr, weight_decay=opts.wd)
        if opts.cycle_lr:
            scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=opts.base_lr, max_lr=opts.max_lr,
                                                    step_size_up=opts.nbatches_train // 2, step_size_down=None,
                                                    mode='triangular', gamma=1.0, scale_fn=None,
                                                    cycle_momentum=False, )

    if not opts.cycle_lr:
        lamda = lambda epoch: opts.learning_rates_mult[epoch]
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lamda)

    return optimizer, scheduler


def Get_learned_params(model, task_id, direction_id):
    """
    For a task_id and direction_id
    Args:
        model: The m
        task_id:
        direction_id

    Returns:

    """
    learned_param = []
    learned_param.extend(model.bumodel.parameters())
    learned_param.extend(model.transfer_learning[task_id])
    return learned_param


def preprocess(inputs: list[torch], device: str) -> torch:
    """
    Args:
        inputs: The list of inputs we desire to move to the device.
        device: The device we desire to transform to.

    Returns: The same inputs but in another device.

    """
    inputs = [inp.to(device) for inp in inputs]  # Moves the tensor into the device, usually to the cuda.
    return inputs
