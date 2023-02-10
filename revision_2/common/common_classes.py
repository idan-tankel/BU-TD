import time
from abc import ABC
from typing import List

import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from common import *
from persons.code.v26.Configs.ConfigClass import ConfigClass


class DatasetInfoBase(ABC):
    measurements = None

    def __init__(self, is_train, ds, nbatches, name, checkpoints_per_epoch=1):
        self.dataset = ds
        self.number_of_batches = nbatches
        self.is_train = is_train
        self.checkpoints_per_epoch = checkpoints_per_epoch
        if self.is_train and checkpoints_per_epoch > 1:
            # when checkpoints_per_epoch>1 we make each epoch smaller
            self.number_of_batches = self.number_of_batches // checkpoints_per_epoch
        self.name = name
        self.dataset_iter = None
        self.needinit = True

    def create_measurement(self, measurements_class, model_opts, model):
        self.measurements = measurements_class(model_opts, model)

    def reset_iter(self):
        self.dataset_iter = iter(self.dataset)

    def do_epoch(self, epoch, opts, number_of_epochs, config: ConfigClass):
        logger.info(self.name)

        nbatches_report = 10
        aborted = False
        cur_batches = 0
        self.measurements.reset()
        if self.needinit or self.checkpoints_per_epoch == 1:
            self.reset_iter()
            self.needinit = False
            if self.is_train and opts.distributed:
                opts.train_sampler.set_epoch(epoch)
                # TODO: when aborted save cur_batches. next, here do for loop and pass over cur_batches
                # and use train_sampler.set_epoch(epoch // checkpoints_per_epoch)
        start_time = time.time()
        return aborted, cur_batches, nbatches_report, start_time


class DataInputs:
    def __init__(self, inputs):
        self.images, self.segmentation, self.label_existence, self.label_all, self.label_task, self.id, self.flag = inputs


class MeasurementsBase:
    """a class for measuring train/test statistics such as accuracy"""
    metrics_cur_batch = None
    cur_batch_size: int = None
    results = None
    n_measurements: int = None
    loss = None
    number_examples: int = None
    metrics = None

    def __init__(self, opts):
        self.opts = opts
        # self.reset()
        self.names = ['Loss']

    def init_results(self):
        self.n_measurements = len(self.names)
        epochs = 1
        self.results = np.full((epochs, self.n_measurements), np.nan)

    # update metrics for current batch and epoch (cumulative)
    # here we update the basic metric (loss). Subclasses should also call update_metric()
    def update(self, inputs, outs, loss):
        cur_batch_size = inputs.images.shape[0]
        self.loss += loss * cur_batch_size
        self.number_examples += cur_batch_size
        self.metrics_cur_batch = [loss * cur_batch_size]
        self.cur_batch_size = cur_batch_size

    # update next metric for current batch
    def update_metric(self, metric, batch_sum):
        self.metrics_cur_batch += [batch_sum]
        metric += batch_sum

    def get_history(self):
        return np.array(self.metrics) / self.number_examples

    # store the epoch's metrics
    def add_history(self, epoch):
        if epoch + 1 > len(self.results):
            more = np.full(((epoch + 1), self.n_measurements), np.nan)  # resize array by 2
            self.results = np.concatenate((self.results, more))
        self.results[epoch, :] = self.get_history()

    def print_data(self, data):
        data_as_string = ''
        for name, metric in zip(self.names, data):
            data_as_string += '{}: {:.2f}, '.format(name, metric)
        data_as_string = data_as_string[:-2]
        return data_as_string

    def print_batch(self):
        data = np.array(self.metrics_cur_batch) / self.cur_batch_size
        return self.print_data(data)

    def print_epoch(self):
        data = self.get_history()
        return self.print_data(data)

    def plot(self, fig, subplots_axes):
        n_measurements = len(self.metrics)
        for measurement_i, name in enumerate(self.names):
            # ax = subplot(1, n_measurements, measurement_i + 1)
            ax = subplots_axes[measurement_i]
            ax.plot(self.results[:, measurement_i])
            ax.set_title(name)

    def reset(self):
        self.loss = np.array(0.0)
        self.number_examples = 0
        self.metrics = [self.loss]

    def add_name(self, name):
        self.names += [name]


class FieldForSavedModel:
    fields_names_for_saved_model: List[str] = ['on_the_run_info', 'training_options']  # TODO - needed?
    on_the_run_info: str = 'on_the_run_info'
    training_options: str = 'training_options'


class DataModule(LightningDataModule):
    def __init__(self, batch_size: int, number_of_workers: int, train_dataset, test_dataset, val_dataset):
        super().__init__()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.number_of_workers = number_of_workers

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.number_of_workers,
                          pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.number_of_workers,
                          shuffle=False,
                          pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.number_of_workers,
                          shuffle=False,
                          pin_memory=True)

    def get_dataloaders(self) -> [DataLoader, DataLoader, DataLoader]:
        return self.train_dataloader(), self.test_dataloader(), self.val_dataloader()
