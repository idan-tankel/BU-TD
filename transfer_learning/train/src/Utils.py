"""
Here we define the function utils.
"""
import argparse
from typing import Iterator, Type, Tuple, List

import torch

import torch.nn as nn

from torch import Tensor

import torch.optim as optim

from .data.Enums import data_set_types, Model_type, TrainingFlag
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from .data.dataset import StanfordCarsDataSet, Food101Dataset, FlowersDataSet, CUB200Dataset, ImageFolderDataSets

import os


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


def create_optimizer_and_scheduler(opts: argparse, learned_params: list) -> \
        [torch.optim, torch.optim.lr_scheduler]:
    """
    Create optimizer and a scheduler according to opts.
    Args:
        opts: The model options.
        learned_params: The learned parameters.

    Returns: Optimizer, scheduler.

    """
    optimizer = None
    if opts.data_set_obj['optimizer'] == 'SGD':
        optimizer = optim.SGD(params=learned_params, lr=opts.data_set_obj['initial_lr'], momentum=0.9,
                              )

    if opts.data_set_obj['optimizer'] == 'Adam':
        optimizer = optim.Adam(params=learned_params,
                               lr=opts.data_set_obj['initial_lr'], weight_decay=opts.data_set_obj['wd'])

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opts.data_set_obj['milestones'] )

    return optimizer, scheduler


def Expand(opts: argparse, mod: Tensor, shape: list) -> Tensor:
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
    mod = torch.nn.functional.interpolate(mod.view((e4, e3, e2, e1)), mode=opts.data_set_obj['interpolation'],
                                          size=(m1, m2))
    e1, e2, e3, e4 = mod.shape
    mod = mod.view((e3, e4, e2, e1))
    return mod


def from_from_data_set_type_to_object(ds_type: data_set_types):
    """
    From data-set layer_type to class object.
    Args:
        ds_type: The data-set layer_type.

    Returns: dataset object.

    """
    if ds_type is data_set_types.StanfordCars:
        return StanfordCarsDataSet
    elif ds_type is data_set_types.Flowers:
        return FlowersDataSet
    elif ds_type is data_set_types.CUB200:
        return CUB200Dataset
    elif ds_type is data_set_types.Food101:
        return Food101Dataset
    else:
        return ImageFolderDataSets


def Get_dataloaders(opts: argparse, task_id: int):
    """
    Get the data-set object.
    Args:
        opts: The model opts.
        task_id: The task id.

    Returns: train, test data-sets.

    """
    ds_type = opts.ds_type
    project_path = opts.project_path
    ntasks = opts.ntasks
    dataset_obj = from_from_data_set_type_to_object(ds_type=opts.ds_type)
    train_ds = dataset_obj(root=os.path.join(project_path, f'data/RAW/{str(ds_type)}'),
                           is_train=True,
                           task=task_id, ntasks=ntasks)
    test_ds = dataset_obj(root=os.path.join(project_path, f'data/RAW/{str(ds_type)}'),
                          is_train=False,
                          task=task_id, ntasks=ntasks)

    train_dl = DataLoader(dataset=train_ds, batch_size=opts.data_set_obj['bs'], shuffle=True, pin_memory=True,
                          num_workers=2)

    test_dl = DataLoader(dataset=test_ds, batch_size=opts.data_set_obj['bs'] * 2, shuffle=False, pin_memory=True,
                         num_workers=2)

    return train_dl, test_dl


def Get_Model_path(opts: argparse, ds_type: data_set_types, training_flag: TrainingFlag,
                   model_type: Model_type, task_id: int) -> Tuple[str, str]:
    """
    Getting the path we store the model into.
    Args:
        opts: The model opts.
        ds_type: The data-set layer_type.
        training_flag: The training flag.
        model_type: The model layer_type.
        task_id: The task id.

    Returns: The path to store the model into.

    """
    opti_type = opts.data_set_obj['optimizer']
    threshold = opts.data_set_obj['threshold']
    bs = opts.data_set_obj['bs']
    milestones = opts.data_set_obj['milestones']
    initial_lr = opts.data_set_obj['initial_lr']
    weight_modulation = opts.data_set_obj['weight_modulation_factor']
    reg = opts.data_set_obj['reg']
    wd = opts.data_set_obj['wd']
    try:
        inter = opts.data_set_obj['interpolation']
    except KeyError:
        inter = ''
    name1 = f'{str(ds_type)}/{str(training_flag)}/{str(model_type)}/' \
            f'{opti_type}/{weight_modulation}/'
    name = name1 + f'Task_{task_id}_threshold_{threshold}_bs_' \
                   f'{bs}_lr=' \
                   f'_{initial_lr}_milestones_' \
                   f'{milestones}_modulation_factor_' \
                   f'{weight_modulation}_interpolation_{inter}_' \
                   f'lambda_{reg}_wd_{wd}'

    return name1, name


def Get_Learned_Params(model: nn.Module, training_flag: TrainingFlag, task_id: int) -> List[nn.Parameter]:
    """
    Get the learned parameters.
    Args:
        model: The model.
        training_flag: The training flag.
        task_id: The task id.

    Returns: list of the learned parameters

    """
    learned_params = []
    if training_flag in [TrainingFlag.Full_Model, TrainingFlag.LWF]:
        learned_params.extend(list(model.parameters()))
    if training_flag is TrainingFlag.Modulation:
        learned_params.extend(model.modulations[task_id])
    if training_flag is TrainingFlag.Classifier_Only:
        learned_params.extend(model.classifier.parameters())
    if training_flag is TrainingFlag.Masks:
        learned_params.extend(model.masks[task_id])

    return learned_params


def Define_Trainer(opts: argparse, name: str) -> pl.Trainer:
    """
    Define the trainer.
    Args:
        opts: The model opts.
        name: The model name.

    Returns: pytorch lightning trainer.

    """
    wandbLogger = WandbLogger(project="Affecting conv weight", job_type='train', name=name,
                              save_dir=os.path.join(opts.project_path, 'data/loggers'))
    checkpoint_second = pl.callbacks.ModelCheckpoint(dirpath=os.path.join(opts.results_dir, name),
                                                     mode='max', monitor='val_acc',
                                                     filename='{val_acc:.3f}')

    trainer = pl.Trainer(max_epochs=opts.data_set_obj['epochs'], accelerator='gpu',
                         logger=wandbLogger, callbacks=[checkpoint_second], precision=16)
    return trainer


def num_params_of_module(model: nn.Module, layer_type: Type[nn.Module]) -> list[nn.Parameter]:
    """
    Computes the parameters of given module type.
    Args:
        model: The model
        layer_type: The mode type.

    Returns: All parameters of given type.

    """
    params = []
    for m in model.modules():
        if isinstance(m, layer_type):
            params.extend(m.parameters())
    return params


def compute_size(size: list, factor: list) -> list:
    """
    Compute size.
    Args:
        size: The size.
        factor: The factor.

    Returns: The modulation size.

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
