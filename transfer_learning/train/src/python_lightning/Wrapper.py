"""
Here we define the model wrapper to support training, validation.
"""
import argparse
import os
import os.path
from typing import Callable, List
from pathlib import Path
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch import Tensor
from ..Utils import create_optimizer_and_scheduler
from ..Modules.Batch_norm import store_running_stats
import torchmetrics
import shutil
from ..data.Enums import TrainingFlag


# Define the model wrapper class.
# Support training and load_model.

class ModelWrapped(LightningModule):
    """
    The Model wrapper class, support initialization, training, testing.
    """

    def __init__(self, opts: argparse, model: nn.Module, learned_params: list,
                 task_id: int, name: str, loss_fun: Callable, training_flag: TrainingFlag):
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
        self.loss_fun: Callable = loss_fun  # The loss criterion.
        # self.dev: str = opts.device  # The device.
        self.opts: argparse = opts  # The model options.
        self.task_id: int = task_id  # The task id.
        self.training_flag = training_flag
        # Define the optimizer, scheduler.
        self.just_initialized = True
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=1000)
        if not os.path.exists(os.path.join(str(Path(__file__).parents[3]), 'data/models', name, 'code')):
            shutil.copytree(str(Path(__file__).parents[2]), os.path.join(str(Path(__file__).parents[3]), 'data/models',
                                                                         name, 'code'))
        print("Code script saved")

    def configure_optimizers(self):
        """
        Returns the optimizer, scheduler.
        """
        optimizer, raw_scheduler = create_optimizer_and_scheduler(opts=self.opts, learned_params=self.learned_params,
                                                                  )
        scheduler = {'scheduler': raw_scheduler, "interval": "epoch"}  # Update the scheduler each step.
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
        if self.training_flag is TrainingFlag.LWF:
            model.eval()  # Move the model into the evaluation mode.
        else:
            model.train()
        _, label, _ = batch
        outs, loss = self.loss_fun(batch, model, 'train')  # Compute the loss.
        class_prob = torch.argmax(outs, dim=1)
        acc = torch.eq(class_prob, label).float()  # Compute the Accuracy.
        acc = acc.mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)  # Update the loss.
        self.log('train_acc', acc, on_step=True, on_epoch=True, sync_dist=True)  # Update the acc.
        if batch_idx == len(self.trainer._data_connector._train_dataloader_source.dataloader()) - 1:
            # store_running_stats(model=self.model, task_id=self.task_id)
            print('Done storing running statistics for BN layers')
        return loss  # Return the loss.

    def test_model(self, batch: List[Tensor], mode: str) -> torch.float:
        """
        Test model.
        Args:
            batch: The batch.
            mode: The mode.

        Returns: The acc

        """
        assert mode in ['test', 'val']
        model = self.model
        model.eval()  # Move the model into the evaluation mode.
        _, label, _ = batch
        with torch.no_grad():  # Without grad.
            # outs = model(images, task_id)  # Forward and make a struct.
            outs, loss = self.loss_fun(batch, model, 'test')  # Compute the loss.
            class_prob = torch.argmax(outs, dim=1)
            acc = torch.eq(class_prob, label).float()  # Compute the Accuracy.
            self.val_acc.update(class_prob, label)
            acc = acc.mean()
            self.log(f'{mode}_loss', loss, on_step=True, on_epoch=True)  # Update the loss.
            self.log(f'{mode}_acc', acc, on_step=True, on_epoch=True)  # Update the acc.
        return acc  # Return the Accuracy and number of inputs in the batch.

    def test_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        """
        Make the validation step.
        Args:
            batch: The inputs.
            batch_idx: The batch id.

        Returns: The task Accuracy on the batch.

        """

        return self.test_model(batch=batch, mode='test')

    def validation_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        """
        Make the validation step.
        Args:
            batch: The inputs.
            batch_idx: The batch id.

        Returns: The task Accuracy on the batch.

        """
        return self.test_model(batch=batch, mode='val')

    def validation_epoch_end(self, outputs: List) -> None:
        """
        The validation epoch end.
        Args:
            outputs: The outputs.

        """
        print(self.val_acc.compute() * 100)
        self.val_acc.reset()
