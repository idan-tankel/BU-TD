"""
The training step.
"""
import os
from pathlib import Path

import pytorch_lightning as pl
import torch.nn.modules as Module
from pytorch_lightning.loggers import WandbLogger

from training.Data.Checkpoints import CheckpointSaver
from training.Data.Get_dataset import get_dataset_for_spatial_relations
from training.Data.Model_Wrapper import ModelWrapped
from training.Data.Structs import Training_flag
from training.Modules.Models import *
from training.Data.Structs import Task_to_struct


def Get_checkpoint_and_logger(opts: argparse, ds_type, task: list[Task_to_struct], epoch: int = 0) -> \
        tuple[CheckpointSaver, WandbLogger, str]:
    """
    Get the checkpoint and logger.
    Args:
        opts: The model opts.
        ds_type: The data-set type.
        task: The task.
        epoch: The epoch.

    Returns:

    """
    project_path = Path(__file__).parents[3]

    if ds_type is DsType.Emnist:
        image_tuple = "(4,4)"
    elif ds_type is DsType.Fashionmnist:
        image_tuple = "(3,3)"
    else:
        image_tuple = "(1,6)"

    data_path = os.path.join(project_path, f'data/{str(ds_type)}/samples/{image_tuple}_Image_Matrix')
    if ds_type is DsType.Omniglot:
        data_path += f"{task[0].task}"
    Checkpoint_saver = CheckpointSaver(
        dirpath=os.path.join(opts.results_dir,
                             f'Model_use_reset_{str(task[0].direction)}_wd_{str(opts.wd)}_base_lr_'
                             f'{opts.base_lr}_max_lr_{opts.max_lr}_'
                             f'epoch_{epoch}_option_bs_{opts.bs}_use_emb_{opts.use_embedding}_ns_{opts.ns}_nfilters_'
                             f'{opts.nfilters}'))
    wandb_logger = WandbLogger(project="My_first_project_5.10", save_dir=opts.results_dir)
    return Checkpoint_saver, wandb_logger, data_path


def train_step(opts: argparse, model: Module, training_flag: Training_flag, task: list[Task_to_struct],
               ds_type: DsType = DsType.Emnist) -> tuple[Module, dict]:
    """
    Train step
    Args:
        opts: The model opts.
        model: The model.
        training_flag: The training flag.
        task: The task we solve.
        ds_type: The data-set type.
    """
    Checkpoint_saver, wandb_logger, data_path = Get_checkpoint_and_logger(opts=opts, ds_type=ds_type, task=task)
    trainer = pl.Trainer(max_epochs=opts.EPOCHS, accelerator='gpu')
    learned_params = training_flag.Get_learned_params(model, task_idx=task[0].task, direction=task[0].direction)
    DataLoaders = get_dataset_for_spatial_relations(opts, data_path, task=task)
    wrapped_model = ModelWrapped(opts, model, learned_params, check_point=Checkpoint_saver,
                                 direction_tuple=task[0].direction,
                                 task_id=task[0].task,
                                 nbatches_train=len(DataLoaders['train_dl']), train_ds=DataLoaders['train_ds'])
    test = False
    if test:
        wrapped_model.load_model(
            model_path='Right_model/BUTDModel_epoch70_direction=(1, 0).pt')
        print(wrapped_model.Accuracy(DataLoaders['test_dl']))
    trainer.fit(wrapped_model, train_dataloaders=DataLoaders['train_dl'], val_dataloaders=DataLoaders['test_dl'])
    return model, DataLoaders
