import numpy as np
import torch

from common.common_classes import MeasurementsBase
from persons.code.v26.models.AutoSimpleNamespace import inputs_to_struct_raw


class Measurements(MeasurementsBase):
    def __init__(self, opts, model):
        super(Measurements, self).__init__(opts)
        # self.reset()
        self.model = model
        self.opts = opts
        if self.opts.use_bu1_loss:
            super().add_name('Occurence Acc')

        if self.opts.use_bu2_loss:
            super().add_name('Task Acc')

        self.init_results()

    def update(self, inputs, outs, loss):
        super().update(inputs, outs, loss)
        outs = get_model_outs(self.model, outs)
        # outs = get_model_outs(model, outs)
        samples = inputs_to_struct_raw(inputs)
        if self.opts.use_bu1_loss:
            occurence_pred = outs.occurence > 0
            occurence_accuracy = (
                    occurence_pred == samples.label_existence).type(
                torch.float).mean(axis=1)
            super().update_metric(self.occurence_accuracy,
                                  occurence_accuracy.sum().cpu().numpy())

        if self.opts.use_bu2_loss:
            preds, task_accuracy = self.opts.task_accuracy(
                outs, samples, self.opts.nclasses)
            super().update_metric(self.task_accuracy,
                                  task_accuracy.sum().cpu().numpy())

    def reset(self):
        super().reset()
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
