import numpy as np
import torch

from v26.models.AutoSimpleNamespace import inputs_to_struct_raw


class MeasurementsBase():
    '''a class for measuring train/test statistics such as accuracy'''

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
    def update(self, inputs, loss):
        """
        update update the loss and the number of examples in the current batch results

        Args:
            inputs (torch.utils.data.dataloader.DataLoader): a list of an objects came from the data loader
            loss (_type_): _description_
            # TODO why do we need all the inputs here? can't we get only the length?
        """        
        cur_batch_size = inputs[1].shape[1]
        self.loss += loss * cur_batch_size
        self.nexamples += cur_batch_size
        self.metrics_cur_batch = [loss * cur_batch_size]
        self.cur_batch_size = cur_batch_size

    # update next metric for current batch
    def update_metric(self, metric, batch_sum):
        self.metrics_cur_batch += [batch_sum]
        metric += batch_sum

    def get_history(self):
        return np.array(self.metrics) / self.nexamples

    # store the epoch's metrics
    def add_history(self, epoch):
        if epoch + 1 > len(self.results):
            more = np.full(((epoch + 1), self.n_measurements), np.nan)  # resize array by 2
            self.results = np.concatenate((self.results, more))
        self.results[epoch, :] = self.get_history()

    def print_data(self, data):
        str = ''
        for name, metric in zip(self.names, data):
            str += '{}: {:.2f}, '.format(name, metric)
        str = str[:-2]
        return str

    def print_batch(self):
        data = np.array(self.metrics_cur_batch) / self.cur_batch_size
        return self.print_data(data)

    def print_epoch(self):
        data = self.get_history()
        return self.print_data(data)

    def plot(self, fig, subplots_axes):
        n_measurements = len(self.metrics)
        for measurementi, name in enumerate(self.names):
            # ax = subplot(1, n_measurements, measurementi + 1)
            ax = subplots_axes[measurementi]
            ax.plot(self.results[:, measurementi])
            ax.set_title(name)

    def reset(self):
        self.loss = np.array(0.0)
        self.nexamples = 0
        self.metrics = [self.loss]

    def add_name(self, name):
        self.names += [name]


class Measurements(MeasurementsBase):
    def __init__(self, opts, model):
        """
        __init__ _summary_

        Args:
            opts (SimpleNamespace): The model opts
            model (): 
        """        
        MeasurementsBase.__init__(self, opts)
        # self.reset()
        self.model = model
        self.opts = opts
        if self.opts.use_bu1_loss:
            MeasurementsBase.add_name('Occurence Acc')

        if self.opts.use_bu2_loss:
            MeasurementsBase.add_name(name='Task Acc',self=self)

        self.init_results()

    def update(self, inputs, outs, loss):
        """
        update update the accuracy for task and occurence tasks of the model
        Using the losses specified in the model opts (config file)
        The measurement class store the model opts and the model itself as it's attributes
        i.e `self.opts`, `self.model`

        Args:
            inputs (_type_): _description_
            outs (_type_): _description_
            loss (_type_): _description_
        """        
        MeasurementsBase.update(self=self, inputs=inputs,loss=loss)
        outs = get_model_outs(self.model, outs)
        # outs = get_model_outs(model, outs)
        samples = inputs_to_struct_raw(inputs)
        if self.opts.use_bu1_loss:
            occurence_pred = outs.occurence > 0
            occurence_accuracy = (
                    occurence_pred == samples.label_existence).type(
                torch.float).mean(axis=1)
            MeasurementsBase.update_metric(self.occurence_accuracy,
                                  occurence_accuracy.sum().cpu().numpy())

        if self.opts.use_bu2_loss:
            preds, task_accuracy = self.opts.task_accuracy(
                outs, samples, self.opts.nclasses)
            MeasurementsBase.update_metric(self.task_accuracy,
                                  task_accuracy.sum().cpu().numpy())

    def reset(self):
        MeasurementsBase.reset(self)
        if self.opts.use_bu1_loss:
            self.occurence_accuracy = np.array(0.0)
            self.metrics += [self.occurence_accuracy]

        if self.opts.use_bu2_loss:
            self.task_accuracy = np.array(0.0)
            self.metrics += [self.task_accuracy]


# %% loss and metrics
def get_model_outs(model, outs):
    if type(model) is torch.nn.DataParallel or type(
            model) is torch.nn.parallel.DistributedDataParallel:
        return model.module.outs_to_struct(outs)
    else:
        return model.outs_to_struct(outs)
