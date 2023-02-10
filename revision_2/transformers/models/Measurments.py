import numpy as np

from common.common_classes import MeasurementsBase


class Measurements(MeasurementsBase):
    def __init__(self, opts, model):
        super().__init__(opts)
        super().add_name('Acc')  # TODO - in the future - slit to task acc, and occurrence acc
        self.accuracy = np.array(0.0)
        self.only_left_acc = np.array(0.0)
        self.metrics = [self.accuracy, self.only_left_acc]
        self.model = model
        self.init_results()

    def init_results(self):
        self.n_measurements = len(self.names)
        epochs = 1  # TODO - what is this?
        self.results = np.full((epochs, self.n_measurements), np.nan)

    def update(self, inputs, outs, loss):
        super().update(inputs, outs, loss)
        preds, task_accuracy, only_left_accuracy = self.opts.task_accuracy(outs, inputs, self.opts.nclasses)
        # super().update_metric(task_accuracy.cpu(), task_accuracy.sum())
        super().update_metric(self.accuracy, task_accuracy.sum().cpu().numpy())
        super().update_metric(self.only_left_acc, only_left_accuracy.sum().cpu().numpy())

    def reset(self):
        super().reset()
        self.accuracy = np.array(0.0)
        self.only_left_acc = np.array(0.0)
        self.metrics += [self.accuracy]
        self.metrics += [self.only_left_acc]
