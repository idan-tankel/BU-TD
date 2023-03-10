"""
The classes handling accuracy, loss and all measurements.
"""

import argparse
from collections import defaultdict
from typing import List, Union

import torch
from avalanche.evaluation import PluginMetric, GenericPluginMetric
from avalanche.evaluation.metrics.accuracy import Accuracy
from avalanche.evaluation.metrics.mean import Mean
from avalanche.training.templates.supervised import SupervisedTemplate
from torch import Tensor

from training.Data.Data_params import Flag
from training.Data.Structs import inputs_to_struct, outs_to_struct
from training.Metrics.Accuracy import multi_label_accuracy_weighted, multi_label_accuracy


class Accuracy_fun(Accuracy):
    """
    Our accuracy class, support multi, weighted accuracy.
    """

    def __init__(self, opts: argparse):

        """
        Args:
            opts: The model opts.
        """
        super(Accuracy_fun, self).__init__()
        # From outs to struct.
        self.out_to_struct = opts.outs_to_struct
        # From inputs to struct.
        self.inputs_to_struct = opts.inputs_to_struct
        # The mean accuracy.
        self._mean_accuracy = defaultdict(Mean)
        # The accuracy function according to the mode.
        if opts.model_flag is Flag.NOFLAG:
            self.accuracy_fun = multi_label_accuracy_weighted
        else:
            self.accuracy_fun = multi_label_accuracy

    @torch.no_grad()
    def update(self, predicted_y: outs_to_struct, gt: inputs_to_struct, task_labels: Union[float, Tensor]) -> None:
        """
        Args:
            predicted_y: The predicted label.
            gt: The ground truth.
            task_labels: The task labels.

        Returns:

        """
        if isinstance(task_labels, int):
            # The predicted 'classes' distribution.
            # The prediction and accuracy.
            preds, task_accuracy = self.accuracy_fun(gt, predicted_y)
            # Update the task accuracy.
            self._mean_accuracy[task_labels].update(task_accuracy, preds.shape[0])

        else:
            raise ValueError(
                f"Task label type: {type(task_labels)}, "
                f"expected int/float or Tensor"
            )


class MyGenericPluginMetric(GenericPluginMetric[float]):
    """
    Base class for all accuracies plugin metrics.
    """

    def __init__(self, opts: argparse, reset_at: str, emit_at: str, mode: str):
        """
        Args:
            opts: The model opts.
            reset_at: When to reset the plugin.
            emit_at: When to show the results.
            mode: Train/val mode.
        """
        # Our accuracy function.
        self._accuracy = Accuracy_fun(opts=opts)
        # Initialize according to the options.
        super(MyGenericPluginMetric, self).__init__(
            self._accuracy, reset_at=reset_at, emit_at=emit_at, mode=mode)

    def update(self, strategy: SupervisedTemplate) -> None:
        """
        Update the plugin.
        Args:
            strategy: The strategy.

        """
        # Update the accuracy plugin.
        self._accuracy.update(strategy.mb_output, strategy.mb_x, 0)


class MinibatchAccuracy(MyGenericPluginMetric):
    """
    The minibatch plugin Accuracy metric.
    This metric only works at training time.

    This metric computes the average Accuracy over patterns
    from a single minibatch.
    It reports the result after each iteration.

    If a more coarse-grained logging is needed, consider using
    class:`EpochAccuracy` instead.
    """

    def __init__(self, opts: argparse):
        """
        Creates an instance of the MinibatchAccuracy metric.
        """
        super(MinibatchAccuracy, self).__init__(opts=opts, reset_at="iteration", emit_at="iteration", mode="train")

    def __str__(self):
        return "Top1_Acc_MB"


class EpochAccuracy(MyGenericPluginMetric):
    """
    The average Accuracy over a single training epoch.
    This plugin metric only works at training time.

    The Accuracy will be logged after each training epoch by computing
    the number of correctly predicted patterns during the epoch divided by
    the overall number of patterns encountered in that epoch.
    """

    def __init__(self, opts: argparse):
        """
        Creates an instance of the EpochAccuracy metric.
        """

        super(EpochAccuracy, self).__init__(opts=opts, reset_at="epoch", emit_at="epoch", mode="train")

    def __str__(self):
        return "Top1_Acc_Epoch"


class RunningEpochAccuracy(MyGenericPluginMetric):
    """
    The average Accuracy across all minibatches up to the current
    epoch iteration.
    This plugin metric only works at training time.

    At each iteration, this metric logs the Accuracy averaged over all patterns
    seen so far in the current epoch.
    The metric resets its state after each training epoch.
    """

    def __init__(self, opts: argparse):
        """
        Creates an instance of the RunningEpochAccuracy metric.
        Args:
            opts: The model opts.
        """
        super(RunningEpochAccuracy, self).__init__(opts=opts, reset_at="epoch", emit_at="iteration", mode="train")

    def __str__(self):
        return "Top1_RunningAcc_Epoch"


class ExperienceAccuracy(MyGenericPluginMetric):
    """
    At the end of each experience, this plugin metric reports
    the average Accuracy over all patterns seen in that experience.
    This metric only works at eval time.
    """

    def __init__(self, opts: argparse):
        """
        Creates an instance of ExperienceAccuracy metric
        Args:
           opts: The model opts.
        """
        super(ExperienceAccuracy, self).__init__(opts=opts, reset_at="experience", emit_at="experience",
                                                 mode="eval")

    def __str__(self):
        return "Top1_Acc_Exp"


class StreamAccuracy(MyGenericPluginMetric):
    """
    At the end of the entire stream of experiences, this plugin metric
    reports the average Accuracy over all patterns seen in all experiences.
    This metric only works at eval time.
    """

    def __init__(self, opts: argparse):
        """
        Creates an instance of StreamAccuracy metric.
        Args:
           opts: The model opts.
        """
        super(StreamAccuracy, self).__init__(opts=opts, reset_at="stream", emit_at="stream", mode="eval")

    def __str__(self):
        return "Top1_Acc_Stream"


def accuracy_metrics(
        opts: argparse,
        minibatch: bool = False,
        epoch: bool = False,
        epoch_running: bool = False,
        experience: bool = False,
        stream: bool = False

) -> List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.
    Args:
        opts: The model opts.
        minibatch: True, will return a metric able to log the minibatch Accuracy at training time.
        epoch: If True, will return a metric able to log the epoch Accuracy at training time.
        epoch_running:  If True, will return a metric able to log the running epoch Accuracy at training time.
        experience: If True, will return a metric able to log the Accuracy on each evaluation experience.
        stream: If True, will return a metric able to log the Accuracy averaged over the entire evaluation stream
        of experiences.

    Returns: A list of plugin metrics.

    """
    metrics = []
    if minibatch:
        metrics.append(MinibatchAccuracy(opts))

    if epoch:
        metrics.append(EpochAccuracy(opts))

    if epoch_running:
        metrics.append(RunningEpochAccuracy(opts))

    if experience:
        metrics.append(ExperienceAccuracy(opts))

    if stream:
        metrics.append(StreamAccuracy(opts))

    return metrics


__all__ = [
    "MinibatchAccuracy",
    "EpochAccuracy",
    "RunningEpochAccuracy",
    "ExperienceAccuracy",
    "StreamAccuracy",
    "accuracy_metrics",
]
