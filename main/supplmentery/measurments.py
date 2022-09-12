import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import argparse
# from ..Configs.Config import Config
from supplmentery.loss_and_accuracy import multi_label_accuracy_base
from supplmentery.get_model_outs import get_model_outs



class MeasurementsBase:
    """
    a class for measuring train/test statistics such as accuracy,loss
    """

    def __init__(self, opts):
        """
        :param opts:
        """
        self.opts = opts
        self.names = ['Loss']  # an attribute(loss) to update.
        epochs = 1
        self.results = np.full((epochs, len(self.names)), np.nan)  # initialize with nan the initial loss

    # update metrics for current batch and epoch (cumulative)
    # here we update the basic metric (loss). Subclasses should also call update_metric()
    def update(self, cur_batch_size: int, loss: float) -> None:
        """
        :param inputs:The input to the model in the current stage.
        :param loss: The loss computed on the batch.
        :return: None
        """
        self.loss += loss * cur_batch_size  # Add to the loss the batch_size * the average loss on the batch size.
        self.nexamples += cur_batch_size  # Add to the number of the seen samples.
        self.metrics_cur_batch = [loss * cur_batch_size]  # Update the current loss.
        self.cur_batch_size = cur_batch_size  # Update the current batch_size

    # update next metric for current batch
    def update_metric(self, metric: np.array, batch_sum: np.array) -> None:
        """
        :param metric: float np.array
        :param batch_sum: float np.array The result of the metric over the batch.
        :return:
        """
        self.metrics_cur_batch += [batch_sum]
        metric += batch_sum

    def get_history(self):
        """
        :return: the average loss until the current stage.
        """
        return np.array(self.metrics) / self.nexamples

    # store the epoch's metrics
    def add_history(self, epoch):
        """
        :param epoch: The epoch number.
        :extends the current history.
        """
        if epoch + 1 > len(self.results):  # If extension is needed and there is no place left.
            more = np.full(((epoch + 1), len(self.names)),
                           np.nan)  # Create an extension of size epoch + 1 - len(self.n_measurements)
            self.results = np.concatenate((self.results, more))  # Add the extension.
        self.results[epoch, :] = self.get_history()  # Add the average loss during the epoch.

    def print_data(self, data: np.array) -> str:
        """
        :param data: Computed metrics
        :return: The metrics in the desired format.
        """
        results_as_dict = dict(zip(self.names, data))
        return results_as_dict

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
    def __init__(self, opts, model: nn.Module) -> None:
        """
        :param opts : Model options.
        :param model :
        """
        MeasurementsBase.__init__(self=self,opts=opts)
        self.model = model
        self.opts = opts
        self.inputs_to_struct = opts.inputs_to_struct
        if self.opts.Losses.use_bu1_loss:  # If desired we also follow the occurrence loss.
            MeasurementsBase.add_name(self=self,name='Occurrence Acc')

        if self.opts.Losses.use_bu2_loss:  # If desired we also follow the task loss.
            MeasurementsBase.add_name(self=self,name='Task Acc')
        # update the number of measurements.
        self.results = np.full((1, len(self.names)), np.nan)

    def update(self, inputs: list[torch.TensorType], outs: list[torch.TensorType], loss: float) -> None:
        """
        update _summary_

        Args:
            inputs (list[torch]): _description_
            outs (list[torch]): The unstractured list of outputs
            loss (float): _description_
        """        
        MeasurementsBase.update(self,cur_batch_size=inputs[0].shape[0], loss=loss)
        outs = get_model_outs(self.model, outs)
        samples = self.inputs_to_struct(inputs)
        if self.opts.Losses.use_bu1_loss:
            occurrence_pred = outs.occurence > 0
            occurrence_accuracy = (occurrence_pred == samples.label_existence).type(torch.float).mean(axis=1)
            MeasurementsBase.update_metric(self=self,metric=self.occurrence_accuracy,
                                  batch_sum=occurrence_accuracy.sum().cpu().numpy())  # Update the occurrence metric.
        if self.opts.Losses.use_bu2_loss:
            preds, task_accuracy = multi_label_accuracy_base(outs, samples)
            MeasurementsBase.update_metric(self,metric=self.task_accuracy, batch_sum=task_accuracy.sum().cpu().numpy())  # Update the task metric.

    def reset(self) -> None:
        """
        :Resets all matrices.
        """
        super().reset()
        if self.opts.Losses.use_bu1_loss:
            self.occurrence_accuracy = np.array(0.0)
            self.metrics += [self.occurrence_accuracy]
        if self.opts.Losses.use_bu2_loss:
            self.task_accuracy = np.array(0.0)
            self.metrics += [self.task_accuracy]


def set_datasets_measurements(datasets: object,model_opts: argparse, model: nn.Module,measurements_class=Measurements):
    """
    set_datasets_measurements Initialize the Measurements class for each dataset and create a Measurements object
    # TODO get rid of that function it's just a wrapper

    Args:
        datasets (List[Dataset]): The list of datasets to initize the Measurements class for. Default = [train,test,valid]
        measurements_class (type): The class to initialize the Measurements class with. Default = Measurements
        model_opts (argparse): The model options object. This is useful to init what losses to keep in mind on...etc using flags
        model (nn.Module): The model to initialize the Measurements class with. Default = None
    """    
    for the_dataset in datasets:
        the_dataset.create_measurement(measurements_class, model_opts, model)
