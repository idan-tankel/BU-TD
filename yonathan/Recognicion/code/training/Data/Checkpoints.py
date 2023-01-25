"""
Here we define the checkpoint class to support
storing of best models.
"""
import argparse
import os
import shutil
from pathlib import Path
from typing import Union, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn


# The Checkpoint class.
# Supports initialization and updates.

class CheckpointSaver:
    """
    Checkpoint class.
    Supports saving models.
    """

    def __init__(self, dirpath: str):
        """
        Checkpoint saver class.
        Saving all needed data.
        Args:
            dirpath: The path to the checkpoint.
            
        """
        self.dirpath = dirpath
        self.optimum = -np.Inf  # initialize with minus infinity.
        # Create the path we save into.
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        code_path = os.path.join(dirpath, 'code')
        # Copy the code script.
        if not os.path.exists(os.path.join(code_path)):
            shutil.copytree(str(Path(__file__).parents[2]), os.path.join(self.dirpath, 'code'))
        print("Code script saved")

    def update_optimum(self, new_optimum: float) -> None:
        """
        Update the optimum with the new optimum.
        Args:
            new_optimum: The new optimum.

        """
        self.optimum = new_optimum

    def __call__(self, model: nn.Module, epoch: int, current_test_accuracy: float, optimizer: torch.optim,
                 scheduler: torch.optim.lr_scheduler, opts: argparse,
                 optional_kay: Optional[Tuple] = None) -> None:
        """
        Saves the state.
        Args:
            model: The model to save.
            epoch: The epoch id.
            current_test_accuracy: The current test Accuracy.
            optimizer: The optimizer.
            scheduler: The scheduler.
            opts: The model options.
            optional_kay: Optional key to add during run-time, needed for baselines.

        """
        # The current model path, updated when new Accuracy is achieved.
        model_path_curr = os.path.join(self.dirpath,
                                       model.__class__.__name__ + f'_epoch{epoch}.pt')
        # The best model path, updated when new Accuracy is achieved.
        model_path_best = os.path.join(self.dirpath,
                                       model.__class__.__name__ + f'_best.pt')
        # The latest model path, updated every epoch.
        model_path_latest = os.path.join(self.dirpath,
                                         model.__class__.__name__ + f'_latest.pt')

        better_than_optimum = current_test_accuracy > self.optimum  # Compute whether we passed the optimum so far.
        # All the data we want to store.
        save_data = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'scheduler_state_dict': scheduler.state_dict(), 'opts': opts}
        if optional_kay is not None:
            (new_key, new_value) = optional_kay
            save_data[new_key] = new_value
        torch.save(save_data, model_path_latest)  # Save the current model in model latest path.
        if epoch % 10 == 0:
            torch.save(save_data, model_path_curr)
        # If we passed the optimum we save in model_id and in model_best.
        if better_than_optimum:
            print(f'New optimum: {current_test_accuracy}, better than {self.optimum}')
            torch.save(save_data, model_path_best)
            torch.save(save_data, model_path_curr)
            self.update_optimum(current_test_accuracy)

        print(f"Current accuracy {current_test_accuracy}")
