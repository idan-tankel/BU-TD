"""
Here we define the baseline utils.
"""
import copy
import os
from typing import Callable

from avalanche.models.utils import avalanche_forward
from avalanche.training.utils import zerolike_params_dict
from torch.utils.data import DataLoader

from training.Data.Data_params import RegType
from training.Data.Parser import GetParser, update_parser
from training.Modules.Create_Models import create_model
from training.Modules.Models import *
from training.Utils import preprocess


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


def construct_flag(opts: argparse, task_id: int, direction_tuple: tuple) -> Tensor:
    """
    Construct new flag from the new task, task id.
    Args:
        opts: The model model_opts.
        task_id: The task id.
        direction_tuple: The task id.

    Returns: The new tasks, with the new task and task.

    """
    # From the task tuple to single number.
    direction_dir, _ = tuple_direction_to_index(num_x_axis=opts.num_x_axis, num_y_axis=opts.num_y_axis,
                                                direction=direction_tuple,
                                                ndirections=opts.ndirections,
                                                task_id=task_id)
    task_id = torch.tensor(task_id)
    # The new task vector.
    New_task_flag = torch.nn.functional.one_hot(task_id, opts.ntasks)
    # The new task vector.
    New_direction_flag = torch.nn.functional.one_hot(direction_dir, opts.ndirections)
    # Concat into one flag.
    New_flag = torch.concat([New_direction_flag, New_task_flag], dim=0).float()
    # Expand into one flag.
    return New_flag.unsqueeze(dim=0)


def set_model(model: nn.Module, state_dict: dict) -> None:
    """
    Set model state by state dict.
    Args:
        model: The model we want to load into.
        state_dict: The state dictionary.

    Returns:

    """
    model.load_state_dict(state_dict=copy.deepcopy(state_dict))


def compute_fisher_information_matrix(opts: argparse, model: nn.Module, criterion: Callable, dataloader: DataLoader,
                                      device: str, train: bool = True, norm=2) -> dict:
    """
    Compute fisher importance matrix for each parameter.
    Args:
        opts: The model model_opts
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
        x = opts.inputs_to_struct(x)  # Make a struct.
        model.zero_grad()  # Reset grads.
        out = avalanche_forward(model, x, task_labels=None)  # Compute output.
        out = opts.outs_to_struct(out)  # Make a struct.
        loss = criterion(opts, x, out)  # Compute the loss.
        loss.backward()  # Compute grads.
        for (k1, p), (k2, imp) in zip(model.feature_extractor.named_parameters(),
                                      importances):  # Iterating over the feature weights.
            assert k1 == k2
            if p.grad is not None:
                # Adding the grad**norm.
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
        # Add the quadratic loss.
        penalty += torch.sum(importance[name] * (param - param_old).pow(2))
        # Update the new loss.
    return penalty


def Norm(opts: argparse, x: inputs_to_struct, out: outs_to_struct) -> torch.float:
    """
    Return the norm of the output classifier.
    Args:
        opts: The model model_opts-not used.
        x: The input struct.
        out: The output struct.

    Returns: The norm of out classifier.

    """
    return torch.norm(out.classifier, dim=1).pow(2).mean()


def Get_updated_opts(ds_type: DsType, reg_type: RegType, model_type=ResNet):
    """
    Args:
        ds_type: The data-set type
        reg_type: The regularization type.
        model_type: The model type.

    Returns:

    """
    opts = GetParser(model_type=model_type, ds_type=ds_type)
    # ResNet to be as large as BU-TD model.
    factor = [2, 1, 1] if ds_type is DsType.Fashionmnist else [3, 3, 3]
    filters = [64, 96, 128, 128] if ds_type is DsType.Fashionmnist else [64, 96, 128, 256]
    update_parser(opts=opts, attr='ns', new_value=factor)  # Make the
    update_parser(opts=opts, attr='use_lateral_bu_td', new_value=False)  # No lateral connections are needed.
    update_parser(opts=opts, attr='use_laterals_td_bu', new_value=False)  # No lateral connections are needed.
    update_parser(opts=opts, attr='nfilters', new_value=filters)
    model = create_model(opts)  # Create the model.
    opts.model = model
    return opts, model


def Get_samples_data(ds_type, lang_id=50):
    """
    Args:
        ds_type:
        lang_id:

    Returns:

    """
    sample_path = ''
    if ds_type is DsType.Emnist:
        sample_path = 'samples/(4,4)_image_matrix'
    elif ds_type is DsType.Fashionmnist:
        sample_path = 'samples/(3,3)_Image_Matrix'
    elif ds_type is DsType.Omniglot:
        sample_path = f'samples/(1,6)_data_set_matrix{lang_id}'
    return sample_path
