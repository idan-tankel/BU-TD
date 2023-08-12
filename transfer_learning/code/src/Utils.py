"""
Here we define the function utils.
"""
import argparse
from typing import Iterator

import torch.optim as optim

from .data.Enums import DataSetTypes, TrainingFlag
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .data.dataset import *

from torchvision.models.mobilenetv2 import MobileNet_V2_Weights


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


def compute_size(size, factor):
    """
    Args:
        size: The filter size.
        factor: The expansion factor.

    Returns: The new filter size(efficient filter).

    """
    (c_in, c_out, k1, k2) = size
    mod1, mod2 = factor
    if c_in < mod1:
        if c_out // (mod1 * mod2) == 0:
            size = []
        else:
            size = (c_in, c_out // (mod1 * mod2), k1, k2)
    elif c_out < mod2:
        if c_in // (mod1 * mod2) == 0:
            size = []
        else:
            size = (c_in // (mod1 * mod2), c_out, k1, k2)
    else:
        size = (c_in // mod1, c_out // mod2, k1, k2)
    return size


def create_optimizer_and_scheduler(opts: argparse, learned_params: list) -> \
        [torch.optim, torch.optim.lr_scheduler]:
    """
    Create optimizer and a scheduler according to opts.
    Args:
        opts: The model options.
        learned_params: The learned parameters.

    Returns: Optimizer, scheduler.

    """
    optimizer = optim.Adam(params=learned_params, lr=opts.task_specification['lr'],
                           weight_decay=opts.task_specification['wd'])

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opts.General_specification['milestones'])

    return optimizer, scheduler


def expand(opts: argparse, mod: Tensor, shape: list) -> Tensor:
    """
    Expand the tensor in interleaved manner to match the neuron's shape.
    Args:
        opts: The model opts.
        mod: The modulations.
        shape: The desired.

    Returns: The expanded modulations.

    """
    e1, e2, e3, e4 = mod.shape
    m1, m2, m3, m4 = shape
    mod = torch.nn.functional.interpolate(mod.view((e4, e3, e2, e1)), mode=opts.task_specification['interpolation'],
                                          size=(m1, m2))
    e1, e2, e3, e4 = mod.shape
    mod = mod.view((e3, e4, e2, e1))
    return mod


def from_data_set_type_to_object(ds_type: DataSetTypes):
    """
    From data-set layer_type to class object.
    Args:
        ds_type: The data-set layer_type.

    Returns: dataset object.

    """
    if ds_type is DataSetTypes.Food101:
        return Food101Dataset
    else:
        return ImageFolderDataSets


def get_dataloaders(opts: argparse, task_id: int):
    """
    Get the data-set object.
    Args:
        opts: The model opts.
        task_id: The task id.

    Returns: train, test data-sets.

    """
    ds_type = opts.ds_type
    project_path = opts.project_path
    num_tasks = opts.General_specification['num_tasks']
    dataset_obj = from_data_set_type_to_object(ds_type=opts.ds_type)
    batch_size = opts.General_specification['bs']
    train_ds = dataset_obj(root=os.path.join(project_path, f'data/RAW/{str(ds_type)}'),
                           is_train=True,
                           task=task_id, num_tasks=num_tasks)
    test_ds = dataset_obj(root=os.path.join(project_path, f'data/RAW/{str(ds_type)}'),
                          is_train=False,
                          task=task_id, num_tasks=num_tasks)

    train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, pin_memory=True,
                          num_workers=2)

    test_dl = DataLoader(dataset=test_ds, batch_size=batch_size * 2, shuffle=False, pin_memory=True,
                         num_workers=2)

    return train_dl, test_dl


def get_model_path(opts: argparse, ds_type: DataSetTypes, training_flag: TrainingFlag) -> str:
    """
    Getting the path we store the model into.
    Args:
        opts: The model opts.
        ds_type: The data-set layer_type.
        training_flag: The training flag.

    Returns: The path to store the model into.

    """
    general_path = f'{str(ds_type)}/{str(training_flag)}/'

    if training_flag is training_flag.Modulation:
        mod_factor = str(opts.task_specification['weight_modulation_factor'])
        inter = opts.task_specification['interpolation']
        general_path = os.path.join(general_path, mod_factor, inter)

    return general_path


def get_learned_params(model: nn.Module, training_flag: TrainingFlag, task_id: int) -> List[nn.Parameter]:
    """
    Get the learned parameters.
    Args:
        model: The model.
        training_flag: The training flag.
        task_id: The task id.

    Returns: list of the learned parameters

    """
    learned_params = []
    if training_flag is TrainingFlag.Full_Model or training_flag is TrainingFlag.LWF:
        learned_params.extend(list(model.parameters()))
    if training_flag is TrainingFlag.Modulation:
        learned_params.extend(model.modulations[task_id])
    if training_flag is TrainingFlag.Masks:
        learned_params.extend(model.masks[task_id])
    if training_flag is TrainingFlag.Classifier_Only:
        learned_params.extend(model.classifier_params[task_id])

    return learned_params


def define_trainer(opts: argparse, name: str) -> pl.Trainer:
    """
    Define the trainer.
    Args:
        opts: The model opts.
        name: The model name.

    Returns: pytorch lightning trainer.

    """
    checkpoint = pl.callbacks.ModelCheckpoint(dirpath=os.path.join(opts.results_dir, name),
                                              mode='max', monitor='val_acc',
                                              filename='{val_acc:.3f}')

    trainer = pl.Trainer(max_epochs=opts.General_specification['epochs'], accelerator='gpu',
                         callbacks=[checkpoint], precision=16,default_root_dir = os.path.join(opts.results_dir, name))
    return trainer


def load_model(model: nn.Module, results_dir: str, model_path: str):
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
    model.load_state_dict(checkpoint)


def load_pretrained_model(model) -> None:
    """
    Load Models.
    Args:
        model: The model.

    Returns: None

    """
    weights = MobileNet_V2_Weights.IMAGENET1K_V2
    old_weights = weights.get_state_dict(progress=True)
    new_check = weights.get_state_dict(progress=True)
    new_checkpoint = dict()
    for name, param in model.state_dict().items():
        if name not in old_weights.keys():
            new_check[name] = param

    for name, param in new_check.items():
        if name in model.state_dict().keys():
            new_checkpoint[name] = param

    for layer in ['weight', 'bias']:
        new_checkpoint[f'classifier.Head.0.{layer}'] = old_weights[f'classifier.1.{layer}']

    model.load_state_dict(new_checkpoint)  # Loading the saved weights.
