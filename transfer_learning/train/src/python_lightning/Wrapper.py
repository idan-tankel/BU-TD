"""
Here we define the model wrapper to support training, validation.
"""
import argparse
import os
import os.path
from typing import Callable, List

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch import Tensor
from src.Utils import create_optimizer_and_scheduler, preprocess
from src.Modules.Batch_norm import store_running_stats
import torchmetrics


# Define the model wrapper class.
# Support training and load_model.

class ModelWrapped(LightningModule):
    """
    The Model wrapper class, support initialization, training, testing.
    """

    def __init__(self, opts: argparse, model: nn.Module, learned_params: list,
                 task_id: int):
        """
        This is the model wrapper for pytorch lightning training.
        Args:
            opts: The model options.
            model: The model.
            learned_params: The parameters we desire to split.
            task_id: The task id.
        """
        super(ModelWrapped, self).__init__()
        self.automatic_optimization = True
        self.need_to_update_running_stats: bool = True
        self.model: nn.Module = model  # The model.
        self.learned_params: list = learned_params  # The learned parameters.
        self.loss_fun: Callable = opts.criterion  # The loss criterion.
        self.dev: str = opts.device  # The device.
        self.opts: argparse = opts  # The model options.
        self.task_id: int = task_id  # The task id.
        # Define the optimizer, scheduler.
        self.just_initialized = True
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=100)

    def configure_optimizers(self):
        """
        Returns the optimizer, scheduler.
        """
        nbatches_train = len(self.trainer._data_connector._train_dataloader_source.dataloader())
        optimizer, raw_scheduler = create_optimizer_and_scheduler(opts=self.opts, learned_params=self.learned_params,
                                                                  nbatches=nbatches_train)
        scheduler = {'scheduler': raw_scheduler, "interval": "step"}  # Update the scheduler each step.
        if raw_scheduler is not None:
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    def training_step(self, batch: List[Tensor], batch_idx: int) -> float:
        """
        Training step.
        Args:
            batch: The inputs.
            batch_idx: The batch id.

        Returns: The loss on the batch.

        """
        model = self.model
        model.train()  # Move the model into the evaluation mode.
        images, label, task_id = batch
        outs = model(images, task_id)  # Forward and make a struct.
        loss = self.loss_fun(outs, label)  # Compute the loss.
        class_prob = torch.argmax(outs, dim=1)
        acc = torch.eq(class_prob, label).float()  # Compute the Accuracy.
        acc = acc.mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)  # Update the loss.
        self.log('train_acc', acc, on_step=True, on_epoch=True, sync_dist=True)  # Update the acc.
        if batch_idx == len(self.trainer._data_connector._train_dataloader_source.dataloader()) - 1:
            store_running_stats(model=self.model, task_id=self.task_id)
            print('Done storing running statistics for BN layers')
        return loss  # Return the loss.

    def validation_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        """
        Make the validation step.
        Args:
            batch: The inputs.
            batch_idx: The batch id.

        Returns: The task Accuracy on the batch.

        """
        return self.test_model(batch=batch,batch_id = batch_idx,  mode='val')

    def test_model(self, batch,batch_id, mode):
        assert mode in ['test', 'val']
        model = self.model
        model.eval()  # Move the model into the evaluation mode.
        images, label, task_id = batch
        with torch.no_grad():  # Without grad.
            outs = model(images, task_id)  # Forward and make a struct.
            loss = self.loss_fun(outs, label)  # Compute the loss.
            class_prob = torch.argmax(outs, dim=1)
            acc = torch.eq(class_prob, label).float()  # Compute the Accuracy.
            self.val_acc.update(class_prob, label)
            acc = acc.mean()
            self.log(f'{mode}_loss', loss, on_step=True, on_epoch=True)  # Update the loss.
            self.log(f'{mode}_acc', acc, on_step=True, on_epoch=True)  # Update the acc.
        if mode == 'val':
            num_batches = len(self.trainer._data_connector._val_dataloader_source.dataloader()) - 1
        else:
            num_batches = len(self.trainer._data_connector._test_dataloader_source.dataloader()) - 1
        if batch_id == num_batches:
            print(self.val_acc.compute())
            self.val_acc.reset()
        return acc  # Return the Accuracy and number of inputs in the batch.

    def test_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        """
        Make the validation step.
        Args:
            batch: The inputs.
            batch_idx: The batch id.

        Returns: The task Accuracy on the batch.

        """

        return self.test_model(batch=batch,batch_id=batch_idx, mode='test')

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
        self.load_state_dict(checkpoint['state_dict'])  # Loading the saved weights.
        if load_opt_and_sche:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint
