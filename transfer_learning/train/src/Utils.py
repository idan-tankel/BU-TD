"""
Here we define the function utils.
"""
import argparse
import os
from typing import Iterator, List, Optional

import numpy as np
import torch.nn as nn

import torch
import torch.optim as optim

from torch import Tensor
from torchvision.models.resnet import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights
from src.data.Enums import data_set_types, Model_type, OptimizerType, ModelFlag, TrainingFlag
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import torch.utils.data as data


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
        [torch.optim, torch.optim.lr_scheduler]:
    """
    Create optimizer and a scheduler according to opts.
    Args:
        opts: The model options.
        learned_params: The learned parameters.
        nbatches: The number of batches in the data-loader

    Returns: Optimizer, scheduler.

    """
    optimizer = None
    if opts.data_set_obj.optimizer_type is OptimizerType.SGD:
        optimizer = optim.SGD(params=learned_params, lr=opts.data_set_obj.initial_lr, momentum=0.9,
                              weight_decay=opts.data_set_obj.wd)

    if opts.data_set_obj.optimizer_type is OptimizerType.Adam:
        optimizer = optim.Adam(params=learned_params, lr=opts.data_set_obj.initial_lr,
                               weight_decay=opts.data_set_obj.wd)

    scheduler = None
    if opts.scheduler_type is optim.lr_scheduler.StepLR:
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=nbatches * 20,
                                              gamma=opts.data_set_obj.factor)
    if opts.scheduler_type is optim.lr_scheduler.CosineAnnealingLR:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    if opts.scheduler_type is optim.lr_scheduler.MultiStepLR:
        milestones = [milestone * nbatches for milestone in opts.data_set_obj.milestones]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones,
                                                   last_epoch=-1, gamma=opts.data_set_obj.factor)

    return optimizer, scheduler


def preprocess(inputs: List[Tensor], device: str) -> List[Tensor]:
    """
    Move list of tensors to the device.
    Args:
        inputs: The list of inputs we desire to move to the device.
        device: The device we desire to transforms to.

    Returns: The same inputs but in another device.

    """
    inputs = [torch.tensor(inp) for inp in inputs]
    inputs = [inp.to(device) for inp in inputs]  # Moves the tensor into the device,
    # usually to the
    # cuda.
    return inputs


def load_model(model: nn.Module, results_dir: str, model_path: str) -> dict:
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
    for name, param in model.state_dict().items():
        if 'modulation' in name and name not in checkpoint['state_dict'].keys():
            checkpoint['state_dict'][name] = param

    model.load_state_dict(checkpoint['state_dict'])  # Loading the saved weights.
    return checkpoint


def Load_pretrained_model(opts: argparse, model: nn.Module, ds_type: data_set_types, model_type: Optional[Model_type]) \
        -> None:
    """
    Load pretrained model.
    Args:
        opts: The model opts.
        model: The model.
        ds_type: The data-set type.
        model_type: The model type.

    """
    checkpoint = {}
    state_dict = {}
    prefix = ''
    if ds_type is not data_set_types.CIFAR100:
        if model_type is Model_type.ResNet18:
            weights = ResNet18_Weights.IMAGENET1K_V1
            num_blocks = [2, 2, 2, 2]
        elif model_type is Model_type.ResNet34:
            weights = ResNet34_Weights.IMAGENET1K_V1
            num_blocks = [3, 4, 6, 3]
        elif model_type is Model_type.ResNet50:
            weights = ResNet50_Weights.IMAGENET1K_V1
            num_blocks = [3, 4, 6, 3]
        else:
            weights = ResNet101_Weights.IMAGENET1K_V1
        state_dict = weights.get_state_dict(progress=True)
    if ds_type is data_set_types.CIFAR100:
        prefix = 'module.'
        model_path = os.path.join(opts.results_dir, 'resnet32.th')
        state_dict = torch.load(model_path)['state_dict']  # Loading the saved data.
    for layer_id in range(len(num_blocks)):
        for block_id in range(num_blocks[layer_id]):
            for layer_type in ['conv1', 'conv2', 'conv3']:
                for param_type in ['weight']:
                    new_key = f'layers.{layer_id}.{block_id}.{layer_type}.{param_type}'
                    old_key = f'{prefix}layer{layer_id + 1}.{block_id}.{layer_type}.{param_type}'
                    try:
                        if new_key in model.state_dict().keys():
                            checkpoint[new_key] = state_dict[old_key]
                    except KeyError:
                        pass
            for layer_type in ['bn1', 'bn2', 'bn3']:
                for param_type in ['weight', 'bias']:
                    new_key = f'layers.{layer_id}.{block_id}.{layer_type}.norm.{param_type}'
                    old_key = f'{prefix}layer{layer_id + 1}.{block_id}.{layer_type}.{param_type}'
                    try:
                        if new_key in model.state_dict().keys():
                            checkpoint[new_key] = state_dict[old_key]
                    except KeyError:
                        pass

            for layer_type in ['downsample.conv1x1']:
                for param_type in ['weight']:
                    new_key = f'layers.{layer_id}.{block_id}.{layer_type}.{param_type}'
                    old_key = f'{prefix}layer{layer_id + 1}.{block_id}.downsample.0.{param_type}'
                    if new_key in model.state_dict().keys():
                        checkpoint[new_key] = state_dict[old_key]

            for layer_type in ['downsample.norm.norm']:
                for param_type in ['weight', 'bias']:
                    new_key = f'layers.{layer_id}.{block_id}.{layer_type}.{param_type}'
                    old_key = f'{prefix}layer{layer_id + 1}.{block_id}.downsample.1.{param_type}'
                    if new_key in model.state_dict().keys():
                        checkpoint[new_key] = state_dict[old_key]

            checkpoint['conv1.weight'] = state_dict[f'{prefix}conv1.weight']
            checkpoint['bn1.norm.weight'] = state_dict[f'{prefix}bn1.weight']
            checkpoint['bn1.norm.bias'] = state_dict[f'{prefix}bn1.bias']

        for key, val in model.state_dict().items():
            if "running_mean" in key or 'running_var' in key or 'linear' in key or 'modulations' in key or \
                    'mod' in key:
                checkpoint[key] = val

    model.load_state_dict(checkpoint)  # Loading the saved weights.


def Expand(mod: Tensor, shapes: list) -> Tensor:
    """
    Expand the tensor in interleaved manner to match the neuron's shape.
    Args:
        mod: The modulations.
        shape: The shape to multiply each dimension.

    Returns: The expanded modulations.

    """
    for dim, shape in enumerate(shapes):
        mod = torch.repeat_interleave(mod, shape, dim=dim)
    return mod


def Change_opts(opts: argparse) -> None:
    """
    Change model opts according to the need.
    Args:
        opts: Model options.

    Returns: None

    """
    layers = opts.data_set_obj.num_blocks
    if opts.model_type is Model_type.ResNet18:
        layers = [2, 2, 2, 2]
    elif opts.model_type is Model_type.ResNet34:
        layers = [3, 4, 6, 3]
    elif opts.model_type is Model_type.ResNet50:
        layers = [3, 4, 6, 3]
    elif opts.model_type is Model_type.ResNet101:
        layers = [3, 4, 23, 3]
    opts.data_set_obj.num_blocks = layers
    if opts.ModelFlag is ModelFlag.Partial_ResNet:
        new_channels = [opts.data_set_obj.channels[i] // max(opts.data_set_obj.weight_modulation_factor) for i in
                        range(len(opts.data_set_obj.channels))]
        opts.channels = new_channels
        opts.weight_modulation = False


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
    ntasks = opts.data_set_obj.ntasks
    train_ds = opts.data_set_obj.dataset_obj(root=os.path.join(project_path, f'data/RAW/{str(ds_type)}'),
                                             is_train=True,
                                             task=task_id, ntasks=ntasks)
    test_ds = opts.data_set_obj.dataset_obj(root=os.path.join(project_path, f'data/RAW/{str(ds_type)}'),
                                            is_train=False,
                                            task=task_id, ntasks=ntasks)
    if ds_type in [data_set_types.CIFAR10, data_set_types.CIFAR100]:
        seed = torch.Generator().manual_seed(0)
        nsamples_train, nsamples_val = int(np.rint(len(train_ds) * 0.9)), int(np.rint(len(train_ds) * 0.1))
        train_ds, val_ds = data.random_split(train_ds, [nsamples_train, nsamples_val], generator=seed)
        train_dl = DataLoader(dataset=train_ds, batch_size=opts.data_set_obj.bs, shuffle=True)
        val_dl = DataLoader(dataset=val_ds, batch_size=opts.data_set_obj.bs * 2, shuffle=False)
        test_dl = DataLoader(dataset=test_ds, batch_size=opts.data_set_obj.bs * 2, shuffle=False)
    else:
        train_dl = DataLoader(dataset=train_ds, batch_size=opts.data_set_obj.bs, shuffle=True)
        val_dl = DataLoader(dataset=test_ds, batch_size=opts.data_set_obj.bs * 2, shuffle=False)
        test_dl = DataLoader(dataset=test_ds, batch_size=opts.data_set_obj.bs * 2, shuffle=False)

    return train_dl, val_dl, test_dl


def Get_Model_path(opts: argparse, ds_type: data_set_types, ModelFlag: ModelFlag, training_flag: TrainingFlag,
                   model_type: Model_type, task_id: int) -> str:
    """
    Getting the path we store the model into.
    Args:
        opts: The model opts.
        ds_type: The data-set type.
        ModelFlag: The Model flag.
        training_flag: The training flag.
        model_type: The model type.
        task_id: The task id.

    Returns: The path to store the model into.

    """
    name = f'{str(ds_type)}/{str(ModelFlag)}/{str(training_flag)}/{str(model_type)}/' \
           f'{str(opts.data_set_obj.optimizer_type)}/'
    name += f'New_{task_id}_factor_{opts.data_set_obj.factor}_wd_{opts.data_set_obj.wd}_bs_' \
            f'{opts.data_set_obj.bs}_lr=' \
            f'_{opts.data_set_obj.initial_lr}_milestones_' \
            f'{opts.data_set_obj.milestones}_drop_out_rate_{opts.data_set_obj.drop_out_rate}_modulation_factor_' \
            f'{opts.data_set_obj.weight_modulation_factor}_channels' \
            f'_{opts.data_set_obj.channels}' \
            f'modulation_{opts.data_set_obj.weight_modulation}'
    return name


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
    if training_flag is TrainingFlag.Full_Model:
        learned_params.extend(list(model.parameters()))
    if training_flag is TrainingFlag.Modulation:
        learned_params.extend(model.modulations[task_id])
    if training_flag is TrainingFlag.Classifier_Only:
        learned_params.extend(model.linear.parameters())
    return learned_params


def Define_Trainer(opts: argparse, name: str) -> pl.Trainer:
    """
    Define the trainer.
    Args:
        opts: The model opts.
        name: The model name.

    Returns: pytorch lightning trainer.

    """
    lightning_logs = os.path.join(opts.project_path, 'data/lightning_logs')
    wandbLogger = WandbLogger(project="Affecting conv weight", job_type='train', name=name,
                              save_dir=os.path.join(opts.project_path, 'data/loggers'))
    checkpoint_second = pl.callbacks.ModelCheckpoint(dirpath=
                                                     os.path.join('/home/sverkip/data/tran/data/models', name),
                                                     save_top_k=1, mode='max', monitor='val_acc', filename=
                                                     'Model_best')
    trainer = pl.Trainer(max_epochs=opts.data_set_obj.epochs, accelerator='gpu',
                         logger=wandbLogger, callbacks=[checkpoint_second])
    return trainer
