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


class ModelWrapped(LightningModule):
    def __init__(self, opts: argparse, model: nn.Module, learned_params: list, check_point: CheckpointSaver,
                 direction_id: tuple[int, int], task_id: int, nbatches_train: int):
        """
        This is the model wrapper for pytorch lightning training.
        Args:
            opts: The parser.
            model: The model.
            learned_params: The parameters we desire to train.
            check_point: The check point.
            direction_id: The direction id.
            task_id: The task id.
            nbatches_train: The number of batches to train.
        """
        super().__init__()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        self.model = model  # The model.
        self.learned_params = learned_params  # The learned parameters.
        self.loss_fun = opts.criterion  # The loss criterion.
        self.accuracy = opts.task_accuracy  # The accuracy criterion.
        self.dev = opts.device  # The device.
        self.opts = opts  # The model options.
        self.check_point = check_point  # The checkpoint saver.
        self.nbatches_train = nbatches_train  # The number of train batches.
        self.direction = direction_id  # The direction id.
        self.task_id = task_id  # The task id.
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
        self.optimizer.zero_grad()  # Reset the optimizer.
        loss.backward()  # Do a backward pass.
        self.optimizer.step()  # Update the model.
        _, acc = self.accuracy(samples, outs)  # The accuracy.
        if type(self.scheduler) in [optim.lr_scheduler.CyclicLR, optim.lr_scheduler.OneCycleLR]:
            self.scheduler.step()  # Make a scheduler step if needed.
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)  # Update loss.
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)  # Update acc.
        return loss  # Return the loss and the output.

    def validation_step(self, batch: list[torch], batch_idx: int) -> float:
        """
        Make the validation step.
        Args:
            batch: The input.
            batch_idx: The batch id.

        Returns: The task accuracy on the batch.

        """

        model = self.model
        model.eval()  # Move the model into the evaluation mode.
        samples = self.opts.inputs_to_struct(batch)
        with torch.no_grad():  # Without grad.
            outs = self.model.forward_and_out_to_struct(samples)  # Forward and make a struct.
            loss = self.loss_fun(self.opts, samples, outs)  # Compute the loss.
            _, acc = self.accuracy(samples, outs)  # Compute the accuracy.
            self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True)  # Update the loss.
            self.log('val_acc', acc, on_step=True, on_epoch=True, logger=True)  # Update the acc.
        return acc.item()

    def configure_optimizers(self) -> tuple[optim, optim.lr_scheduler]:
        """
        Returns the optimizer, scheduler.
        """
        opti, scheduler = create_optimizer_and_scheduler(self.opts, self.learned_params, self.nbatches_train)
        return [opti], [scheduler]

    def validation_epoch_end(self, outputs: list) -> float:
        """
        Args:
            outputs: All accuracy outputs.

        Returns: The overall accuracy.

        """
        self.optimizer.zero_grad(
            set_to_none=True)  # Set to None for not used parameters in order to avoid moving of not used weights.
        acc = sum(outputs) / len(outputs)  # The accuracy.
        if self.check_point is not None:  # Update the checkpoint.
            self.check_point(self.model, self.current_epoch, acc, self.optimizer, self.scheduler, self.opts,
                             self.task_id, self.direction)
        return acc

    def accuracy_dl(self, dl: DataLoader) -> float:
        """
        Args:
            dl: The data loader we desire to test.

        Returns: The overall accuracy over the batch.

        """
        acc = 0.0
        self.model.eval()  # Move into evaluation mode.
        for inputs in dl:  # Iterating over all inputs.
            inputs = preprocess(inputs, device=self.dev)  # Move to the device.
            samples = self.opts.inputs_to_struct(inputs)  # From input to Samples.
            outs = self.model.forward_and_out_to_struct(samples)  # Forward and making a struct.
            _, acc_batch = self.accuracy(samples, outs)  # Compute the accuracy.
            acc += acc_batch
        acc = acc / len(dl)  # Normalize to [0,1].
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
