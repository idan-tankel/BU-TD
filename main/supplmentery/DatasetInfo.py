import argparse
import logging
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class DatasetInfo:
    """encapsulates a (train/test/validation) dataset with its appropriate train or test function and Measurement class"""

    def __init__(self, istrain: bool, data_set: DataLoader, nbatches: int, name: str, checkpoints_per_epoch: int = 1,
                 sampler=None):
        """
        Args:
            istrain: Whether we should fit the dataset.
            data_set:  The data set.
            nbatches: Number of batches in the data_set.
            name: The dataset name.
            checkpoints_per_epoch: Number of checkpoints in the epoch..
            sampler: The sampler.
        """
        self.dataset = data_set
        self.nbatches = nbatches
        self.istrain = istrain
        self.checkpoints_per_epoch = checkpoints_per_epoch
        if self.istrain and checkpoints_per_epoch > 1:
            self.nbatches = self.nbatches // checkpoints_per_epoch
        if istrain:  # If we fit the data_loader we choose each step to be train_step including backward step,schedular step.
            self.batch_fun = train_step
        else:  # otherwise we just do a forward pass and don't update the model and the scheduler.
            self.batch_fun = test_step
        self.name = name
        self.dataset_iter = None
        self.needinit = True
        self.sampler = sampler

    def create_measurement(self, measurements_class: type, opts: argparse, model: nn.Module):
        """
        We create measurment object to handle our matrices.
        Args:
            measurements_class: The measurement class should handle the desired matrices.
            opts: The model options.
            model: The model.

        """
        self.measurements = measurements_class(opts, model)

    def reset_iter(self):
        """
        crate a dataset iterator.
        """
        self.dataset_iter = iter(self.dataset)

    def do_epoch(self, opts: argparse, epoch: int) -> None:
        """
        Args:
            opts: The model options.
            epoch: The epoch id.

        """
        opts.logger.info(self.name)
        nbatches_report = 10
        aborted = False
        self.measurements.reset()  # Reset the measurements class.
        cur_batches = 0
        if self.needinit or self.checkpoints_per_epoch == 1:
            self.reset_iter()  # Reset the data loader to iterate from the beginning.
            self.needinit = False  # The initialization is done.
        start_time = time.time()  # Count the beginning time.
        for inputs in self.dataset_iter:  # Iterating over all dataset.
            cur_loss, outs = self.batch_fun(opts, inputs)  # compute the model outputs and the current loss.
            with torch.no_grad():
                # so that accuracies calculation will not accumulate gradients
                self.measurements.update(inputs, outs, cur_loss.item())  # update the loss and the accuracy.
            cur_batches += 1  # Update the number of batches.
            template = 'Epoch {} step {}/{} {} ({:.1f} estimated minutes/epoch)'  # Define a convenient template.
            if cur_batches % nbatches_report == 0:
                duration = time.time() - start_time  # Compute the step time.
                start_time = time.time()
                estimated_epoch_minutes = duration / 60 * self.nbatches / nbatches_report  # compute the proportion time.
                opts.logger.info(template.format(epoch + 1, cur_batches, self.nbatches, self.measurements.print_batch(),
                                                 estimated_epoch_minutes))  # Add the epoch_id, loss, accuracy, time for epoch.

        if not aborted:
            self.needinit = True
        self.measurements.add_history(epoch)  # Update the loss and accuracy history.
