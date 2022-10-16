import argparse

import numpy as np
import torch
import torch.nn as nn

from supp.general_functions import get_model_outs


class MeasurementsBase:
    """
    a class for measuring train/test statistics such as accuracy,loss
    """

    def __init__(self, opts):
        """
        Args:
            opts: The model options.
        """
        self.opts = opts
        self.names = ['Loss']  # an attribute(loss) to update.

    def init_results(self):
        """
        :Initialize the class.
        """
        self.n_measurements = len(self.names)
        epochs = 1
        self.results = np.full((epochs, self.n_measurements), np.nan)  # initialize with nan the initial loss

    # update metrics for current batch and epoch (cumulative)
    # here we update the basic metric (loss). Subclasses should also call update_metric()
    def update(self, inputs: list[torch], outs: list[torch], loss: float):
        """
        Args:
            inputs: input to the model in the current stage.
            outs: The model outs.
            loss: The loss computed on the batch.

        """

        cur_batch_size = inputs[0].shape[0]  # Getting the batch size as the first dimension.
        self.loss += loss * cur_batch_size  # Add to the loss the batch_size * the average loss on the batch size.
        self.nexamples += cur_batch_size  # Add to the number of the seen samples.
        self.metrics_cur_batch = [loss * cur_batch_size]  # Update the current loss.
        self.cur_batch_size = cur_batch_size  # Update the current batch_size

    # update next metric for current batch
    def update_metric(self, metric: np.array, batch_sum: np.array):
        """
        Args:
            metric: The metric we want to update.
            batch_sum: The batch sum.

        """
        self.metrics_cur_batch += [batch_sum]
        metric += batch_sum

    def get_history(self):
        """
        Returns the average loss until the current stage.
        """
        return np.array(self.metrics) / self.nexamples

    # store the epoch's metrics
    def add_history(self, epoch: int):
        """
        Extends the current history.
        Args:
            epoch: The epoch number.

        """

        if epoch + 1 > len(self.results):  # If extension is needed and there is no place left.
            more = np.full(((epoch + 1), self.n_measurements),
                           np.nan)  # Create an extension of size epoch + 1 - len(self.n_measurements)
            self.results = np.concatenate((self.results, more))  # Add the extension.
        self.results[epoch, :] = self.get_history()  # Add the average loss during the epoch.

    def print_data(self, data: np.array) -> str:
        """
        :param data: Computed metrics
        :return: The metrics in the desired format.
        """
        data_str = ''
        for name, metric in zip(self.names, data):  # For each metric name we concatenate the name with its value.
            data_str += '{}: {:.2f}, '.format(name, metric)
        data_str = data_str[:-2]
        return data_str

    def print_batch(self):
        """
        :return: the average matrices computed in the current batch.
        """
        data = np.array(self.metrics_cur_batch) / self.cur_batch_size
        return self.print_data(data)  # Use the previous method to return the appropriate string.

    def print_epoch(self):
        """
        :return: A string of the average matrices over all epoch.
        """
        data = self.get_history()  # Get the average matrices computed during the epoch.
        return self.print_data(data)  # return the string according to the format.

    def plot(self, subplots_axes):
        """
        Plotting the current metres.
        :param subplots_axes:
        :return:
        """
        for measurementi, name in enumerate(self.names):
            ax = subplots_axes[measurementi]
            ax.plot(self.results[:, measurementi])
            ax.set_title(name)

    def reset(self) -> None:
        """
        :resets all matrices.
        """
        self.loss = np.array(0.0)
        self.nexamples = 0
        self.metrics = [self.loss]

    def add_name(self, name: str) -> None:
        """
        Adding a name to the metres.
        :param name: The desired name.
        :return:
        """
        self.names += [name]


class Measurements(MeasurementsBase):
    def __init__(self, opts: argparse, model: nn.Module) -> None:
        """
        :param opts : Options
        :param model :
        """
        super(Measurements, self).__init__(opts)
        self.model = model
        self.opts = opts
        self.inputs_to_struct = opts.inputs_to_struct
        if self.opts.use_bu1_loss:  # If desired we also follow the occurrence loss.
            super().add_name('Occurrence Acc')

        if self.opts.use_bu2_loss:  # If desired we also follow the task loss.
            super().add_name('Task Acc')

        self.init_results()  # Initialize the matrices.

    def update(self, inputs: list[torch], outs: list[torch], loss: float) -> None:
        super().update(inputs, outs, loss)
        outs = get_model_outs(self.model, outs)
        samples = self.inputs_to_struct(inputs)
        if self.opts.use_bu1_loss:
            occurrence_pred = outs.occurence_out > 0
            occurrence_accuracy = (occurrence_pred == samples.label_existence).type(torch.float).mean(axis=1)
            super().update_metric(self.occurrence_accuracy,
                                  occurrence_accuracy.sum().cpu().numpy())  # Update the occurrence metric.
        if self.opts.use_bu2_loss:
            preds, task_accuracy = self.opts.task_accuracy(outs, samples)
            super().update_metric(self.task_accuracy, 10 * task_accuracy.sum().cpu().numpy())  # Update the task metric.

    def reset(self) -> None:
        """
        :Resets all matrices.
        """
        super().reset()
        if self.opts.use_bu1_loss:
            self.occurrence_accuracy = np.array(0.0)
            self.metrics += [self.occurrence_accuracy]
        if self.opts.use_bu2_loss:
            self.task_accuracy = np.array(0.0)
            self.metrics += [self.task_accuracy]


def set_datasets_measurements(datasets: object, measurements_class: type, model_opts: argparse, model: nn.Module):
    for the_dataset in datasets:
        the_dataset.create_measurement(measurements_class, model_opts, model)
