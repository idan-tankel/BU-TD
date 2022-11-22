import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from training.Modules.Batch_norm import store_running_stats


# The Checkpoint class.

class CheckpointSaver:
    def __init__(self, dirpath: str, store_running_statistics: bool = False):
        """
        Checkpoint saver class.
        Saving all needed data.
        Args:
            dirpath: The path to the checkpoint.
            store_running_statistics: Whether to store the running stats.
            
        """
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        self.dirpath = dirpath
        self.best_metric_val = -np.Inf
        code_path = os.path.join(dirpath, 'code')
        self.store_running_stats = store_running_statistics
        if not os.path.exists(os.path.join(code_path)):
            shutil.copytree(Path(__file__).parents[2], os.path.join(self.dirpath, 'code'))

    def __call__(self, model: nn.Module, epoch: int, metric_val: float, optimizer: torch.optim,
                 scheduler: torch.optim.lr_scheduler, parser: argparse, task_id: int, direction: tuple):
        """
        Saves the state.
        Args:
            model: The model to save.
            epoch: The epoch id.
            metric_val: The current value.
            optimizer: The optimizer.
            scheduler: The scheduler.
            parser: The parser.
            task_id: The task id.
            direction: The direction id.

        """
        model_path_curr = os.path.join(self.dirpath,
                                       model.__class__.__name__ + f'_epoch{epoch}_direction={direction}.pt')
        model_path_best = os.path.join(self.dirpath,
                                       model.__class__.__name__ + '_best_direction={}.pt'.format(direction))
        model_path_latest = os.path.join(self.dirpath,
                                         model.__class__.__name__ + '_latest_direction={}.pt'.format(direction))

        better_than_optimum = metric_val > self.best_metric_val
        if self.store_running_stats:
            store_running_stats(model, task_id=task_id, direction_id=direction)
            print('Done storing running stats')
        self.best_metric_val = metric_val
        save_data = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'scheduler_state_dict': scheduler.state_dict(), 'parser': parser}
        torch.save(save_data, model_path_latest)
     #   torch.save(save_data, model_path_curr)
        if better_than_optimum:
            print('New optimum: {}, better than {}'.format(metric_val, self.best_metric_val))
            torch.save(save_data, model_path_best)
            torch.save(save_data, model_path_curr)


def load_model(model: nn.Module, results_path: str, model_path: str) -> dict:
    """
     Loads and returns the model checkpoint as a dictionary.
    Args:
        model: The model we want load to.
        results_path: The path to the result dir.
        model_path: The path to the model.

    Returns: Load the state to the model and returns the checkpoint.

    """
    model_path = os.path.join(results_path, model_path)
    checkpoint = torch.load(model_path)  # Loading the weights and the metadata.
    model.load_state_dict(checkpoint['model_state_dict'])  # Loading the state dict.
    return checkpoint
