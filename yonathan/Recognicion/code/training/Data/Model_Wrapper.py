"""
Here we define the model wrapper to support training, validation.
"""
import argparse
import os
import os.path

import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

from training.Data.Checkpoints import CheckpointSaver
from training.Utils import create_optimizer_and_scheduler, preprocess

from typing import Callable

from training.Modules.Batch_norm import store_running_stats

from training.Data.Data_params import Flag, DsType

from torch.utils.data import Dataset


# Define the model wrapper class.
# Support training and load_model.

class ModelWrapped(LightningModule):
    """
    The Model wrapper class, support initialization, training, testing.
    """

    def __init__(self, opts: argparse, model: nn.Module, learned_params: list, check_point: CheckpointSaver,
                 direction_tuple: list[tuple[int, int]], task_id: int, nbatches_train: int, train_ds: Dataset):
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
        self.learned_params: list = learned_params  # The learned parameters.
        self.loss_fun: Callable = opts.criterion  # The loss criterion.
        self.accuracy: Callable = opts.task_accuracy  # The Accuracy criterion.
        self.dev: str = opts.device  # The device.
        self.opts: argparse = opts  # The model options.
        self.check_point: CheckpointSaver = check_point  # The checkpoint saver.
        self.nbatches_train: int = nbatches_train  # The number of train batches.
        self.direction: list[tuple[int, int]] = direction_tuple  # The direction id.
        self.task_id: int = task_id  # The task id.
        self.store_running_stats: bool = self.opts.model_flag is Flag.CL
        self.ds_type: DsType = self.opts.ds_type
        # Define the optimizer, scheduler.
        self.optimizer, self.scheduler = create_optimizer_and_scheduler(self.opts, self.learned_params,
                                                                        self.nbatches_train)
        self.train_ds: Dataset = train_ds  # Our data-set.

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
        samples = self.opts.inputs_to_struct(inputs=batch)  # Compute the sample struct.
        outs = model.forward_and_out_to_struct(inputs=samples)  # Compute the model output.
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
            store_running_stats(model=self.model, task_id=self.task_id, direction_id=self.direction[0])
            print('Done storing running stats')

        model = self.model
        model.eval()  # Move the model into the evaluation mode.
        samples = self.opts.inputs_to_struct(inputs=batch)
        with torch.no_grad():  # Without grad.
            outs = self.model.forward_and_out_to_struct(inputs=samples)  # Forward and make a struct.
            loss = self.loss_fun(opts=self.opts, samples=samples, outs=outs)  # Compute the loss.
            _, acc = self.accuracy(samples=samples, outs=outs)  # Compute the Accuracy.
            self.log('val_loss', loss, on_step=True, on_epoch=True)  # Update the loss.
            self.log('val_acc', acc, on_step=True, on_epoch=True)  # Update the acc.
        return acc, batch[0].size(0)  # Return the Accuracy and number of samples in the batch.

    def configure_optimizers(self) -> tuple[optim, optim.lr_scheduler]:
        """
        Returns the optimizer, scheduler.
        """
        optimizer, scheduler = create_optimizer_and_scheduler(opts=self.opts, learned_params=self.learned_params,
                                                              nbatches=self.nbatches_train)
        scheduler = {'scheduler': scheduler, "interval": "step"}  # Update the scheduler each step.
        return [optimizer], [scheduler]

    def validation_epoch_end(self, outputs: list[tuple]) -> float:
        """
        Args:
            outputs: All Accuracy outputs.

        Returns: The overall Accuracy.

        """
        # Set to None for not used parameters in order to avoid moving of not used weights.
        self.optimizer.zero_grad(set_to_none=True)
        num_successes = sum(outputs[i][0] * outputs[i][1] for i in range(len(outputs)))
        num_images = sum(outputs[i][1] for i in range(len(outputs)))
        acc = num_successes / num_images  # The Accuracy.
        # Update the checkpoint.
        if self.check_point is not None:
            self.check_point(model=self.model, epoch=self.current_epoch, current_test_accuracy=acc,
                             optimizer=self.optimizer, scheduler=self.scheduler,
                             opts=self.opts, task_id=self.task_id, direction=self.direction[0])
        print(f"Final accuracy {acc}")
        return acc

    def train_dataloader(self):
        """
        We have observed that during training, resting the train dataloader shows much better result
        for EMNIST, Fashion-MNIST.
        """

        dataloader = DataLoader(dataset=self.train_ds, batch_size=self.opts.bs, num_workers=self.opts.workers,
                                shuffle=True, pin_memory=True)
        return dataloader

    def training_epoch_end(self, outputs: list) -> None:
        """
        After each epoch we reset the data-loader for EMNIST, Fashion-MNIST.
        """
        if self.ds_type is not DsType.Omniglot:
            self.train_dataloader()

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
