"""
Here we define the model wrapper to support training, validation.
"""
import argparse
import os
import os.path
from typing import Callable

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from .Checkpoints import CheckpointSaver
from .Data_params import Flag, DsType
from ..Modules.Batch_norm import store_running_stats
from ..Utils import create_optimizer_and_scheduler, preprocess
from ..Data.Structs import inputs_to_struct, outs_to_struct

# Define the model wrapper class.
# Support training and load_model.

class ModelWrapped(LightningModule):
    """
    The Model wrapper class, support initialization, training, testing.
    """

    def __init__(self, opts: argparse, model: nn.Module, learned_params: list, check_point: CheckpointSaver,
                 direction_tuple: tuple[int, int], task_id: int, nbatches_train: int, train_ds: Dataset):
        """
        This is the model wrapper for pytorch lightning training.
        Args:
            opts: The model options.
            model: The model.
            learned_params: The parameters we desire to train.
            check_point: The check point.
            direction_tuple: The direction id.
            task_id: The task id.
            nbatches_train: The number of batches to train.
            train_ds: train ds, for reloading the data-loader epoch.
        """
        super(ModelWrapped, self).__init__()
        self.automatic_optimization = True
        self.need_to_update_running_stats: bool = True
        self.model: nn.Module = model  # The model.
        self.learned_params: list[nn.Parameter] = learned_params  # The learned parameters.
        self.loss_fun: Callable = opts.criterion  # The loss criterion.
        self.accuracy: Callable = opts.data_obj.task_accuracy  # The Accuracy criterion.
        self.dev: str = opts.device  # The device.
        self.opts: argparse = opts  # The model options.
        self.check_point: CheckpointSaver = check_point  # The checkpoint saver.
        self.nbatches_train: int = nbatches_train  # The number of train batches.
        self.direction: tuple[int, int] = direction_tuple  # The direction id.
        self.task_id: int = task_id  # The task id.
        self.store_running_stats: bool = self.opts.model_flag is Flag.CL
        self.ds_type: DsType = self.opts.ds_type
        # Define the optimizer, scheduler.
     #   self.optimizer, self.scheduler = create_optimizer_and_scheduler(self.opts, self.learned_params,
       #                                                                 self.nbatches_train)
        self.train_ds: Dataset = train_ds  # Our data-set.
    
    def configure_optimizers(self):
        """
        Returns the optimizer, scheduler.
        """
        optimizer, scheduler = create_optimizer_and_scheduler(opts=self.opts, learned_params=self.learned_params,
                                                              nbatches=self.nbatches_train)
        if scheduler is not None:
            scheduler = {'scheduler': scheduler, "interval": "epoch"}  # Update the scheduler each step.
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    def training_step(self, batch: list[Tensor], batch_idx: int) -> float:
        """
        Training step.
        Args:
            batch: The input.
            batch_idx: The batch id.

        Returns: The loss on the batch.

        """
        model = self.model
        model.train()  # Move the model into the train mode.
        samples = inputs_to_struct(inputs=batch)  # Compute the sample struct.
        outs = model(samples)  # Compute the model output.
        outs = outs_to_struct(outs)
      #  print(self.scheduler)
        loss = self.loss_fun(opts=self.opts, samples=samples, outs=outs)  # Compute the loss.
        _, acc = self.accuracy(samples=samples, outs=outs)  # The Accuracy.
        self.log('train_loss', loss, on_step=True, on_epoch=True)  # Update loss.
        self.log('train_acc', acc, on_step=True, on_epoch=True)  # Update acc.
        return loss  # Return the loss.

    def validation_step(self, batch: list[Tensor], batch_idx: int) -> tuple[float, float]:
        """
        Make the validation step.
        Args:
            batch: The input.
            batch_idx: The batch id.

        Returns: The task Accuracy on the batch.

        """
        if batch_idx == 0 and self.need_to_update_running_stats and self.store_running_stats:
            store_running_stats(model=self.model, task_id=self.task_id, direction_id=self.direction)
            print('Done storing running stats')

        model = self.model
        model.eval()  # Move the model into the evaluation mode.
        samples = inputs_to_struct(inputs=batch)
        with torch.no_grad():  # Without grad.
            outs = self.model(samples)  # Forward and make a struct.
            outs = outs_to_struct(outs)
            loss = self.loss_fun(opts=self.opts, samples=samples, outs=outs)  # Compute the loss.
            _, acc = self.accuracy(samples=samples, outs=outs)  # Compute the Accuracy.
            self.log('val_loss', loss, on_step=True, on_epoch=True)  # Update the loss.
            self.log('val_acc', acc, on_step=True, on_epoch=True)  # Update the acc.
        return acc, batch[0].size(0)  # Return the Accuracy and number of inputs in the batch.



    def validation_epoch_end(self, outputs: list[tuple]) -> float:
        """
        Args:
            outputs: All Accuracy outputs.

        Returns: The overall Accuracy.

        """
        num_successes = sum(outputs[i][0] * outputs[i][1] for i in range(len(outputs)))
        num_images = sum(outputs[i][1] for i in range(len(outputs)))
        acc = num_successes / num_images  # The Accuracy.
        # Update the checkpoint.
        if self.check_point is not None:
            self.check_point(model=self.model, epoch=self.current_epoch, current_test_accuracy=acc,
                             opts=self.opts)
        print(f"Final accuracy {acc}")
        return acc

    def Accuracy(self, dl: DataLoader) -> float:
        """
        Args:
            dl: The data loader we desire to test.

        Returns: The overall Accuracy over the data-set.

        """
        acc = 0.0
        num_inputs = 0.0
        self.need_to_update_running_stats = False
        self.model.eval()  # Move into evaluation mode.
        for inputs in dl:  # Iterating over all inputs.
            inputs = preprocess(inputs=inputs, device=self.dev)  # Move to the device.
            acc_batch, batch_size = self.validation_step(batch=inputs, batch_idx=0)  # The accuracy.
            acc += acc_batch * batch_size  # Add to the Accuracy
            num_inputs += batch_size
        self.need_to_update_running_stats = True
        acc /= num_inputs  # Normalize to [0,1].
        return acc

    def load_model(self, model_path: str, load_opt_and_sche: bool = False) -> dict:
        """
        Loads and returns the model checkpoint as a dictionary.
        Args:
            model_path: The path to the model.
            load_opt_and_sche: Whether to load also the optimizer, scheduler state.

        Returns: The loaded checkpoint.

        """
        results_path = self.opts.results_dir  # Getting the result dir.
        model_path = os.path.join(results_path, model_path)  # The path to the model.
        checkpoint = torch.load(model_path)  # Loading the saved data.
        self.model.load_state_dict(checkpoint['model_state_dict'])  # Loading the saved weights.
        if load_opt_and_sche:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint
