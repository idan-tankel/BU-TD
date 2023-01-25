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
from training.Data.Structs import Task_to_struct
from training.Data.Structs import Training_flag
from training.Metrics.Accuracy import accuracy
from training.Modules.Models import *


def Get_checkpoint_and_logger(opts: argparse, ds_type, task: list[Task_to_struct], epoch: int = 0) -> \
        tuple[CheckpointSaver, WandbLogger, str]:
    """
    Get the checkpoint and logger.
    Args:
        opts: The model opts.
        ds_type: The data-set type.
        task: The task.
        epoch: The epoch.

    Returns: The Checkpoint saver, wandb logger, sample path.

    """
    # The project path.
    project_path = str(Path(__file__).parents[3])
    # Results dir.
    results_dir = opts.results_dir
    # The sample path.
    sample_path = Get_sample_path(project_path=project_path, ds_type=ds_type, task=task)
    # The checkpoint-saver.
    Checkpoint_saver = CheckpointSaver(
        dirpath=os.path.join(results_dir,
                             f'Model_use_reset_{str(task[0].direction)}_wd_{str(opts.wd)}_base_lr_'
                             f'{opts.base_lr}_max_lr_{opts.max_lr}_'
                             f'epoch_{epoch}_option_bs_{opts.bs}_use_emb_{opts.use_embedding}_ns_{opts.ns}_nfilters_'
                             f'{opts.nfilters}_initial_tasks_{opts.initial_directions[0].unified_task}'))
    #  Wandb logger.
    wandb_logger = WandbLogger(project="Training_Continual_Learning", save_dir=results_dir)
    return Checkpoint_saver, wandb_logger, sample_path


def Get_sample_path(project_path: str, ds_type: DsType, task: list[Task_to_struct]) -> str:
    """
    Get the sample path of the data-set type.
    Args:
        project_path: The project path.
        ds_type: The data-set type.
        task: The task(needed for Omniglot)

    Returns: The sample path.

    """
    if ds_type is DsType.Emnist:
        image_tuple = "(4,4)"
    elif ds_type is DsType.Fashionmnist:
        image_tuple = "(3,3)"
    else:
        image_tuple = "(1,6)"
    # The file format.
    data_path = os.path.join(project_path, f'data/{str(ds_type)}/samples/{image_tuple}_Image_Matrix')
    if ds_type is DsType.Omniglot:
        data_path += f"{task[0].task}"
    return data_path


def train_step(opts: argparse, model: Module, training_flag: Training_flag, task: list[Task_to_struct],
               ds_type, epochs: int) -> tuple[Module, dict]:
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
    Checkpoint_saver, wandb_logger, sample_path = Get_checkpoint_and_logger(opts=opts, ds_type=ds_type, task=task)
    # The trainer.
    trainer = pl.Trainer(max_epochs=epochs, logger=wandb_logger, accelerator='gpu')
    # Get the learned-params according to the training-flag.
    learned_params = training_flag.Get_learned_params(model, task_idx=task[0].task, direction=task[0].direction)
    # The data dictionary.
    DataLoaders = get_dataset_for_spatial_relations(opts, sample_path, task=task)
    # The wrapped model.
    wrapped_model = ModelWrapped(opts, model, learned_params, check_point=Checkpoint_saver,
                                 direction_tuple=task[0].direction,
                                 task_id=task[0].task,
                                 nbatches_train=len(DataLoaders['train_dl']), train_ds=DataLoaders['train_ds'])
    # Train the model.
    trainer.fit(wrapped_model, train_dataloaders=DataLoaders['train_dl'], val_dataloaders=DataLoaders['test_dl'])
    # Return the trained model, and the data-loaders.
    return model, DataLoaders


def Show_zero_forgetting(opts: argparse, models: list[nn.Module], data: list[dict]) -> list[list]:
    """
    Args:
        opts: The model opts.
        models: The models.
        data: The data-loaders dictionary.

    Returns: The accuracy for all appropriate model and data.

    """
    # The number of tasks.
    num_tasks = len(models)
    # The accuracy list.
    Accuracy = [[] for _ in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(num_tasks):
            # The model is trained on that data.
            if j < i + 1:
                # The model.
                model = models[j]
                # The data-loader.
                test_data_loader = data[j]
                # The accuracy.
                acc = accuracy(opts, model, test_data_loader['test_dl'])
                Accuracy[i].append(acc)
            else:
                # The model is not trained on that data.
                Accuracy[i].append("*")
    return Accuracy
