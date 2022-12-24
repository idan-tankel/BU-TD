import argparse
import os
import os.path

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

from training.Data.Checkpoints import CheckpointSaver
from training.Utils import create_optimizer_and_scheduler, preprocess

from typing import Callable


# Define the model wrapper class.
# Support training and load_model.

class ModelWrapped(LightningModule):
    """
    The Model wrapper class, support initialization, training, testing.
    """

    def __init__(self, opts: argparse, model: nn.Module, learned_params: list, check_point: CheckpointSaver,
                 direction_tuple: tuple[int, int], task_id: int, nbatches_train: int):
        """
        This is the model wrapper for pytorch lightning training.
        Args:
            opts: The parser.
            model: The model.
            learned_params: The parameters we desire to train.
            check_point: The check point.
            direction_tuple: The direction id.
            task_id: The task id.
            nbatches_train: The number of batches to train.
        """
        super().__init__()
        self.automatic_optimization = True
        self.model: nn.Module = model  # The model.
        self.learned_params: list = learned_params  # The learned parameters.
        self.loss_fun: Callable = opts.criterion  # The loss criterion.
        self.accuracy: Callable = opts.task_accuracy  # The Accuracy criterion.
        self.dev: str = opts.device  # The device.
        self.opts: argparse = opts  # The model options.
        self.check_point: CheckpointSaver = check_point  # The checkpoint saver.
        self.nbatches_train: int = nbatches_train  # The number of train batches.
        self.direction: tuple[int, int] = direction_tuple  # The direction id.
        self.task_id: int = task_id  # The task id.
        # Define the optimizer, scheduler.
        self.optimizer, self.scheduler = create_optimizer_and_scheduler(self.opts, self.learned_params,
                                                                        self.nbatches_train)

    def training_step(self, batch: list[torch], batch_idx: int) -> float:
        """
        Training step.
        Args:
            batch: The input.
            batch_idx: The batch id.

        Returns: The loss on the batch.

        """
        model = self.model
        model.train()  # Move the model into the train mode.
        samples = self.opts.inputs_to_struct(batch)  # Compute the sample struct.
        outs = model.forward_and_out_to_struct(samples)  # Compute the model output.
        loss = self.loss_fun(self.opts, samples, outs)  # Compute the loss.
        _, acc = self.accuracy(samples, outs)  # The Accuracy.
        self.log('train_loss', loss, on_step=True, on_epoch=True)  # Update loss.
        self.log('train_acc', acc, on_step=True, on_epoch=True)  # Update acc.
        return loss  # Return the loss.

    def validation_step(self, batch: list[torch], batch_idx: int) -> float:
        """
        Make the validation step.
        Args:
            batch: The input.
            batch_idx: The batch id.

        Returns: The task Accuracy on the batch.

        """

        model = self.model
        model.eval()  # Move the model into the evaluation mode.
        samples = self.opts.inputs_to_struct(batch)
        with torch.no_grad():  # Without grad.
            outs = self.model.forward_and_out_to_struct(samples)  # Forward and make a struct.
            loss = self.loss_fun(self.opts, samples, outs)  # Compute the loss.
            _, acc = self.accuracy(samples, outs)  # Compute the Accuracy.
            self.log('val_loss', loss, on_step=True, on_epoch=True)  # Update the loss.
            self.log('val_acc', acc, on_step=True, on_epoch=True)  # Update the acc.
        return acc  # Return the Accuracy.

    def configure_optimizers(self) -> tuple[optim, optim.lr_scheduler]:
        """
        Returns the optimizer, scheduler.
        """
        optimizer, scheduler = create_optimizer_and_scheduler(self.opts, self.learned_params, self.nbatches_train)
        scheduler = {'scheduler': scheduler, "interval": "step"}  # Update the scheduler each step.
        return [optimizer], [scheduler]

    def validation_epoch_end(self, outputs: list) -> float:
        """
        Args:
            outputs: All Accuracy outputs.

        Returns: The overall Accuracy.

        """
        self.optimizer.zero_grad(
            set_to_none=True)  # Set to None for not used parameters in order to avoid moving of not used weights.
        acc = sum(outputs) / len(outputs)  # The Accuracy.
        # Update the checkpoint.
        if self.check_point is not None:
            self.check_point(self.model, self.current_epoch, acc, self.optimizer, self.scheduler, self.opts,
                             self.task_id, self.direction)
        return acc

    def Accuracy(self, dl: DataLoader) -> float:
        """
        Args:
            dl: The data loader we desire to test.

        Returns: The overall Accuracy over the data-set.

        """
        acc = 0.0
        self.model.eval()  # Move into evaluation mode.
        for inputs in dl:  # Iterating over all inputs.
            inputs = preprocess(inputs, device=self.dev)  # Move to the device.
            samples = self.opts.inputs_to_struct(inputs)  # From input to Samples.
            outs = self.model.forward_and_out_to_struct(samples)  # Forward and making a struct.
            _, acc_batch = self.accuracy(samples, outs)  # Compute the Accuracy.
            acc += acc_batch  # Add to the Accuracy
        acc /= len(dl)  # Normalize to [0,1].
        return acc

    def load_model(self, model_path: str) -> dict:
        """
        Loads and returns the model checkpoint as a dictionary.
        Args:
            model_path: The path to the model.

        Returns: The loaded checkpoint.

        """
        results_path = self.opts.results_dir  # Getting the result dir.
        model_path = os.path.join(results_path, model_path)  # The path to the model.
        checkpoint = torch.load(model_path)  # Loading the saved data.
        self.model.load_state_dict(checkpoint['model_state_dict'])  # Loading the saved weights.
        return checkpoint
