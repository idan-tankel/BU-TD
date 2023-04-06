"""
The training step.
"""
import os
from pathlib import Path

import pytorch_lightning as pl
import torch.nn.modules as Module
from pytorch_lightning.loggers import WandbLogger
from ..pytorch_lightning.lightning_model import ModelWrapped
from typing import Tuple
import torch.nn as nn
from torch.utils.data import DataLoader
from ..Utils import load_pretrained_model, num_params, Get_Learned_Params, Get_sample_path, Create_Model_name
import argparse
from Data_Creation.src.Create_dataset_classes import DsType
import torch
from ..Data.Enums import Training_type, Flag
from ..Data.datasets.Datasets import DatasetGuidedSingleTask, DatasetNonGuided
from ..Data.datasets.classification_datasets import *


def from_data_type_to_dataset(ds_type: DsType, flag_at: Flag):
    if ds_type is DsType.Emnist or ds_type is DsType.Fashionmnist:
        if flag_at is Flag.CL:
            return DatasetGuidedSingleTask
        else:
            return DatasetNonGuided

    if ds_type is DsType.Food:
        return Food101Dataset

    if ds_type is DsType.EmnistSingle:
        return EmnistSingleDataSet


def Get_dataloaders(opts, root, ds_type, flag_at, task):
    dataset = from_data_type_to_dataset(ds_type=ds_type, flag_at=flag_at)
    train_ds = dataset(root=root, opts=opts, is_train=True, task=task)
    test_ds = dataset(root=root, opts=opts, is_train=False, task=task)
    batch_size = opts.data_obj['bs']
    train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, num_workers=opts.data_obj['workers'], shuffle=True,
                          pin_memory=True)  # The Train Data-Loader.
    test_dl = DataLoader(dataset=test_ds, batch_size=batch_size, num_workers=opts.data_obj['workers'], shuffle=False,
                         pin_memory=True)  # The Test Data-Loader.
    return train_dl, test_dl


def train_step(opts: argparse, model: Module, training_flag: Training_type, task: Tuple[int, Tuple],
               ds_type: DsType, epochs: int, flag_at: Flag) -> tuple[Module, dict]:
    """
    Train step
    Args:
        opts: The model opts.
        model: The model.
        training_flag: The training flag.
        task: The task we solve.
        ds_type: The data-set type.
        epochs: The number of epochs.
    """
    # The Checkpoint-saver, wandb-logger, the sample path.
    project_path = opts.project_path
    results_dir = os.path.join(project_path,
                               f'data/{str(ds_type)}/results/')
    name = Create_Model_name(opts,task=task[-1])
    wandb_logger = WandbLogger(project="Training_Continual_Learning", name=name, save_dir=results_dir, job_type='train')
    sample_path = Get_sample_path(project_path=project_path, ds_type=ds_type, task=task)
    #
    checkpoint = pl.callbacks.ModelCheckpoint(dirpath=os.path.join(results_dir, name),
                                              mode='max', monitor='val_acc',
                                              filename='{val_acc:.3f}')
    #
    # The trainer.
    path = os.path.join(os.path.join(project_path,
                               f'data/{str(DsType.Emnist)}/results/'),
                                        'initial_lr_0.01_wd_0.0_milestones_[]_scheduler_None_drop_out_rate_0.0_factor_0.1_optimizer_Adam_heads_[1000, 196, 200, 102, 101]_'
                                        '/val_acc=0.952.ckpt')

    trainer = pl.Trainer(max_epochs=epochs, logger=wandb_logger, accelerator='gpu', callbacks=[checkpoint], )
    # Get the learned-params according to the training-flag.
    learned_params = Get_Learned_Params(opts, model, task_id=task, training_flag=training_flag, ds_type=ds_type)
    # The data dictionary.
    train_dl, test_dl = Get_dataloaders(opts=opts, ds_type=ds_type, flag_at=flag_at, root=sample_path, task=task)
    # The wrapped model.
    wrapped_model = ModelWrapped(opts, model, learned_params, task_id=task, name=name)
    wrapped_model.load_state_dict(torch.load(path)['state_dict'])
    # Train the model.
    trainer.fit(wrapped_model, train_dataloaders=train_dl, val_dataloaders=test_dl)
    # Return the trained model, and the data-loaders.
    return model, (train_dl, test_dl)
