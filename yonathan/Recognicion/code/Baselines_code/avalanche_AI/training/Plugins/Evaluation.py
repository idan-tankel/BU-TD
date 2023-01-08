################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 30-12-2020                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

from collections import defaultdict
from typing import List, Union, Dict

import torch
from avalanche.evaluation import Metric, PluginMetric, GenericPluginMetric
from avalanche.evaluation.metric_utils import phase_and_task
from avalanche.evaluation.metrics.mean import Mean
from torch import Tensor
from training.Data.Data_params import Flag
from training.Metrics.Accuracy import multi_label_accuracy_weighted, multi_label_accuracy
from avalanche.evaluation.metrics.accuracy import Accuracy

from Baselines_code.avalanche_AI.training.supervised.strategy_wrappers import SupervisedTemplate

import argparse


class Accuracy_fun(Accuracy):

    def __init__(self, parser: argparse):
        """
        Args:
            parser: The model parser.
        """
        # From outs to struct.
        self.out_to_struct = parser.outs_to_struct
        # From inputs to struct.
        self.inputs_to_struct = parser.inputs_to_struct
        # The mean accuracy.
        self._mean_accuracy = defaultdict(Mean)
        # The accuracy function according to the mode.
        if parser.model_flag is Flag.NOFLAG:
            self.accuracy_fun = multi_label_accuracy_weighted
        else:
            self.accuracy_fun = multi_label_accuracy

    @torch.no_grad()
    def update(self, predicted_y: Tensor, gt: Tensor, task_labels: Union[float, Tensor]) -> None:
        """
        Args:
            predicted_y: The predicted label.
            gt: The ground truth.
            task_labels: The task labels.

        Returns:

        """
        if isinstance(task_labels, int):
            # The predicted 'classes' distribution.
            predicted_y_struct = self.out_to_struct(predicted_y)
            # The prediction and accuracy.
            preds, task_accuracy = self.accuracy_fun(gt, predicted_y_struct)
            # Update the task accuracy.
            self._mean_accuracy[task_labels].update(task_accuracy, preds.shape[0])

        elif isinstance(task_labels, Tensor):
            for pred, true, t in zip(predicted_y, gt[1], task_labels):
                true_positives = (pred == true).float().item()
                self._mean_accuracy[t.item()].update(true_positives, 1)
        else:
            raise ValueError(
                f"Task label type: {type(task_labels)}, "
                f"expected int/float or Tensor"
            )

    def result(self, task_label=None) -> Dict[int, float]:
        """
        Return the current results.
        Args:
            task_label: The task label.

        Returns: The accuracy per task.

        """
        assert task_label is None or isinstance(task_label, int)
        if task_label is None:
            return {k: v.result() for k, v in self._mean_accuracy.items()}
        else:
            return {task_label: self._mean_accuracy[task_label].result()}

    def reset(self, task_label=None):
        """
        Resets the accuracy metrics.
        Args:
            task_label: if None, reset the entire dictionary.
            Otherwise, reset the value associated to `task_label`.

        """
        assert task_label is None or isinstance(task_label, int)
        if task_label is None:
            self._mean_accuracy = defaultdict(Mean)
        else:
            self._mean_accuracy[task_label].reset()


class MyGenericPluginMetric(GenericPluginMetric[float]):
    """
    Base class for all accuracies plugin metrics.
    """

    def __init__(self, parser: argparse, reset_at: str, emit_at: str, mode: str):
        """
        Args:
            parser: The parser.
            reset_at: When to reset the plugin.
            emit_at: When to show the results.
            mode: Train/val mode.
        """
        # Our accuracy function.
        self._accuracy = Accuracy_fun(parser)
        # Initialize according to the options.
        super(MyGenericPluginMetric, self).__init__(
            self._accuracy, reset_at=reset_at, emit_at=emit_at, mode=mode
        )

    def reset(self, strategy: Union[SupervisedTemplate, None] = None) -> None:
        """
        Reset the plugin.
        Args:
            strategy: The strategy.

        """
        # Reset the plugin.
        if self._reset_at == "stream" or strategy is None:
            self._metric.reset()
        else:
            self._metric.reset(phase_and_task(strategy)[1])

    def result(self, strategy: Union[SupervisedTemplate, None] = None) -> float:
        """
        Return the result of the plugin.
        Args:
            strategy: The strategy.

        Returns:

        """
        # return the result.
        if self._emit_at == "stream" or strategy is None:
            return self._metric.result()
        else:
            return self._metric.result(phase_and_task(strategy)[1])

    def update(self, strategy: SupervisedTemplate) -> None:
        """
        Update the plugin.
        Args:
            strategy: The strategy.

        Returns:

        """
        # Update the accuracy plugin.

        # Task labels defined for each experience.
        if hasattr(strategy.experience, "task_labels"):
            task_labels = strategy.experience.task_labels
        else:
            task_labels = [0]  # add fixed task label if not available.

        if len(task_labels) > 1:
            # task labels defined for each pattern
            task_labels = strategy.mb_task_id
        else:
            task_labels = task_labels[0]
        self._accuracy.update(strategy.mb_output, strategy.mb_x, task_labels)


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

    def __init__(self, parser: argparse):
        """
        Creates an instance of the MinibatchAccuracy metric.
        """
        super(MinibatchAccuracy, self).__init__(parser=parser, reset_at="iteration", emit_at="iteration", mode="train")

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

    def __init__(self, parser: argparse):
        """
        Creates an instance of the EpochAccuracy metric.
        """

        super(EpochAccuracy, self).__init__(parser=parser, reset_at="epoch", emit_at="epoch", mode="train")

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

    def __init__(self, parser: argparse):
        """
        Creates an instance of the RunningEpochAccuracy metric.
        Args:
            parser: The model opts.
        """
        super(RunningEpochAccuracy, self).__init__(parser=parser, reset_at="epoch", emit_at="iteration", mode="train")

    def __str__(self):
        return "Top1_RunningAcc_Epoch"


class ExperienceAccuracy(MyGenericPluginMetric):
    """
    At the end of each experience, this plugin metric reports
    the average Accuracy over all patterns seen in that experience.
    This metric only works at eval time.
    """

    def __init__(self, parser: argparse):
        """
        Creates an instance of ExperienceAccuracy metric
        Args:
           parser: The model opts.
        """
        super(ExperienceAccuracy, self).__init__(parser=parser, reset_at="experience", emit_at="experience",
                                                 mode="eval")

    def __str__(self):
        return "Top1_Acc_Exp"


class StreamAccuracy(MyGenericPluginMetric):
    """
    At the end of the entire stream of experiences, this plugin metric
    reports the average Accuracy over all patterns seen in all experiences.
    This metric only works at eval time.
    """

    def __init__(self, parser: argparse):
        """
        Creates an instance of StreamAccuracy metric.
        Args:
           parser: The model opts.
        """
        super(StreamAccuracy, self).__init__(parser=parser, reset_at="stream", emit_at="stream", mode="eval")

    def __str__(self):
        return "Top1_Acc_Stream"


def accuracy_metrics(
        parser: argparse,
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
        parser: The model opts.
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
        metrics.append(MinibatchAccuracy(parser))

    if epoch:
        metrics.append(EpochAccuracy(parser))

    if epoch_running:
        metrics.append(RunningEpochAccuracy(parser))

    if experience:
        metrics.append(ExperienceAccuracy(parser))

    if stream:
        metrics.append(StreamAccuracy(parser))

    return metrics


__all__ = [
    "MinibatchAccuracy",
    "EpochAccuracy",
    "RunningEpochAccuracy",
    "ExperienceAccuracy",
    "StreamAccuracy",
    "accuracy_metrics",
]
