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
from typing import List, Union, Dict
import torch
from torch import Tensor
from avalanche.evaluation import Metric, PluginMetric, GenericPluginMetric
from avalanche.evaluation.metrics.mean import Mean
from avalanche.evaluation.metric_utils import phase_and_task
from collections import defaultdict
from supp.data_functions import dev

def multi_label_accuracy_base(outs: object, samples: object) -> tuple:
    """
    The base class for all modes.
    Here for each head we compute its accuracy according to the model out and label task.
    Args:
        outs: The model outs.
        samples: The samples.

    Returns: The predictions, the task accuracy.

    """
    predictions = torch.argmax(outs, dim=1)
    label_task = samples[1]
    task_accuracy = ( predictions == label_task).float()  # Compare the number of matches and normalize by the batch size*num_outputs.
    return (predictions, task_accuracy)

def multi_label_accuracy(outs: object, samples: object):
    """
    return the task accuracy mean over all samples.
    Args:
        outs: The model outs.
        samples: The samples.

    Returns: The predictions and task accuracy over the batch.

    """
    preds, task_accuracy = multi_label_accuracy_base(outs, samples)
    task_accuracy = task_accuracy.mean(axis=0)  # per single example
    return preds, task_accuracy

def multi_label_accuracy_weighted(outs, inputs):
    """
    return the task accuracy weighted mean over the existing characters in the image.
    Args:
        outs: The model outs.
        samples: The samples.

    Returns: The predication and mean accuracy over the batch.

    """
    preds, task_accuracy = multi_label_accuracy_base(outs, inputs)
    loss_weight = inputs[-1]
    task_accuracy = task_accuracy * loss_weight
    task_accuracy = task_accuracy.sum() / loss_weight.sum()  # per single example
    return preds, task_accuracy

class MyAccuracy(Metric[float]):
    """
    The Accuracy metric. This is a standalone metric.

    The metric keeps a dictionary of <task_label, accuracy value> pairs.
    and update the values through a running average over multiple
    <prediction, target> pairs of Tensors, provided incrementally.
    The "prediction" and "target" tensors may contain plain labels or
    one-hot/logit vectors.

    Each time `result` is called, this metric emits the average accuracy
    across all predictions made since the last `reset`.

    The reset method will bring the metric to its initial state. By default
    this metric in its initial state will return an accuracy value of 0.
    """

    def __init__(self,accuracy_name):
        """
        Creates an instance of the standalone Accuracy metric.

        By default this metric in its initial state will return an accuracy
        value of 0. The metric can be updated by using the `update` method
        while the running accuracy can be retrieved using the `result` method.
        """
        self._mean_accuracy = defaultdict(Mean)
        if accuracy_name == 'MACW':
         self.accuracy_fun = multi_label_accuracy_weighted
        else:
         self.accuracy_fun = multi_label_accuracy_base
        """
        The mean utility that will be used to store the running accuracy
        for each task label.
        """

    @torch.no_grad()
    def update(self, predicted_y: Tensor, gt: Tensor, task_labels: Union[float, Tensor],) -> None:
        """
        Update the running accuracy given the true and predicted labels.
        Parameter `task_labels` is used to decide how to update the inner
        dictionary: if Float, only the dictionary value related to that task
        is updated. If Tensor, all the dictionary elements belonging to the
        task labels will be updated.

        :param predicted_y: The model prediction. Both labels and logit vectors
            are supported.
        :param true_y: The ground truth. Both labels and one-hot vectors
            are supported.
        :param task_labels: the int task label associated to the current
            experience or the task labels vector showing the task label
            for each pattern.

        :return: None.
        """
        if isinstance(task_labels, int):
            #

            preds, task_accuracy = self.accuracy_fun(predicted_y, gt)
            #
          #  true_positives = float(torch.sum(torch.eq(predicted_y, true_y)))
          #  total_patterns = len(true_y)
            task_accuracy = task_accuracy.sum()
            self._mean_accuracy[task_labels].update(task_accuracy , predicted_y.shape[0])

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
        Retrieves the running accuracy.

        Calling this method will not change the internal state of the metric.

        :param task_label: if None, return the entire dictionary of accuracies
            for each task. Otherwise return the dictionary
            `{task_label: accuracy}`.
        :return: A dict of running accuracies for each task label,
            where each value is a float value between 0 and 1.
        """
        assert task_label is None or isinstance(task_label, int)
        if task_label is None:
            return {k: v.result() for k, v in self._mean_accuracy.items()}
        else:
            return {task_label: self._mean_accuracy[task_label].result()}

    def reset(self, task_label=None) -> None:
        """
        Resets the metric.
        :param task_label: if None, reset the entire dictionary.
            Otherwise, reset the value associated to `task_label`.

        :return: None.
        """
        assert task_label is None or isinstance(task_label, int)
        if task_label is None:
            self._mean_accuracy = defaultdict(Mean)
        else:
            self._mean_accuracy[task_label].reset()


class MyAccuracyPluginMetric(GenericPluginMetric[float]):
    """
    Base class for all accuracies plugin metrics
    """

    def __init__(self, reset_at, emit_at, mode,accuracy_fun):
        self._accuracy = MyAccuracy(accuracy_fun)
        super(MyAccuracyPluginMetric, self).__init__(
            self._accuracy, reset_at=reset_at, emit_at=emit_at, mode=mode
        )

    def reset(self, strategy=None) -> None:
        if self._reset_at == "stream" or strategy is None:
            self._metric.reset()
        else:
            self._metric.reset(phase_and_task(strategy)[1])

    def result(self, strategy=None) -> float:
        if self._emit_at == "stream" or strategy is None:
            return self._metric.result()
        else:
            return self._metric.result(phase_and_task(strategy)[1])

    def update(self, strategy):
        # task labels defined for each experience
        if hasattr(strategy.experience, "task_labels"):
            task_labels = strategy.experience.task_labels
        else:
            task_labels = [0]  # add fixed task label if not available.

        if len(task_labels) > 1:
            # task labels defined for each pattern
            task_labels = strategy.mb_task_id
        else:
            task_labels = task_labels[0]
        self._accuracy.update(strategy.mb_output[0], strategy.mb_x, task_labels)


class MinibatchAccuracy(MyAccuracyPluginMetric):
    """
    The minibatch plugin accuracy metric.
    This metric only works at training time.

    This metric computes the average accuracy over patterns
    from a single minibatch.
    It reports the result after each iteration.

    If a more coarse-grained logging is needed, consider using
    :class:`EpochAccuracy` instead.
    """

    def __init__(self,accuracy_fun):
        """
        Creates an instance of the MinibatchAccuracy metric.
        """
        super(MinibatchAccuracy, self).__init__(
            reset_at="iteration", emit_at="iteration", mode="train",accuracy_fun = accuracy_fun
        )

    def __str__(self):
        return "Top1_Acc_MB"


class EpochAccuracy(MyAccuracyPluginMetric):
    """
    The average accuracy over a single training epoch.
    This plugin metric only works at training time.

    The accuracy will be logged after each training epoch by computing
    the number of correctly predicted patterns during the epoch divided by
    the overall number of patterns encountered in that epoch.
    """

    def __init__(self,accuracy_metrics=None):
        """
        Creates an instance of the EpochAccuracy metric.
        """

        super(EpochAccuracy, self).__init__(
            reset_at="epoch", emit_at="epoch", mode="train",accuracy_fun=accuracy_metrics
        )

    def __str__(self):
        return "Top1_Acc_Epoch"


class RunningEpochAccuracy(MyAccuracyPluginMetric):
    """
    The average accuracy across all minibatches up to the current
    epoch iteration.
    This plugin metric only works at training time.

    At each iteration, this metric logs the accuracy averaged over all patterns
    seen so far in the current epoch.
    The metric resets its state after each training epoch.
    """

    def __init__(self):
        """
        Creates an instance of the RunningEpochAccuracy metric.
        """

        super(RunningEpochAccuracy, self).__init__(
            reset_at="epoch", emit_at="iteration", mode="train"
        )

    def __str__(self):
        return "Top1_RunningAcc_Epoch"


class ExperienceAccuracy(MyAccuracyPluginMetric):
    """
    At the end of each experience, this plugin metric reports
    the average accuracy over all patterns seen in that experience.
    This metric only works at eval time.
    """

    def __init__(self, accuracy_fun):
        """
        Creates an instance of ExperienceAccuracy metric
        """
        super(ExperienceAccuracy, self).__init__(
            reset_at="experience", emit_at="experience", mode="eval",accuracy_fun=accuracy_fun
        )

    def __str__(self):
        return "Top1_Acc_Exp"

class StreamAccuracy(MyAccuracyPluginMetric):
    """
    At the end of the entire stream of experiences, this plugin metric
    reports the average accuracy over all patterns seen in all experiences.
    This metric only works at eval time.
    """

    def __init__(self,accuracy_fun):
        """
        Creates an instance of StreamAccuracy metric
        """
        super(StreamAccuracy, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval",accuracy_fun=accuracy_fun
        )

    def __str__(self):
        return "Top1_Acc_Stream"


class TrainedExperienceAccuracy(MyAccuracyPluginMetric):
    """
    At the end of each experience, this plugin metric reports the average
    accuracy for only the experiences that the model has been trained on so far.

    This metric only works at eval time.
    """

    def __init__(self,accuracy_fun):
        """
        Creates an instance of TrainedExperienceAccuracy metric by first
        constructing AccuracyPluginMetric
        """
        super(TrainedExperienceAccuracy, self).__init__(
            reset_at="stream", emit_at="stream", mode="eval",accuracy_fun =accuracy_fun
        )
        self._current_experience = 0

    def after_training_exp(self, strategy) -> None:
        self._current_experience = strategy.experience.current_experience
        # Reset average after learning from a new experience
        MyAccuracyPluginMetric.reset(self, strategy)
        return MyAccuracyPluginMetric.after_training_exp(self, strategy)

    def update(self, strategy):
        """
        Only update the accuracy with results from experiences that have been
        trained on
        """
        if strategy.experience.current_experience <= self._current_experience:
            MyAccuracyPluginMetric.update(self, strategy)

    def __str__(self):
        return "Accuracy_On_Trained_Experiences"


def accuracy_metrics(
    accuracy_fun,
    minibatch=False,
    epoch=False,
    epoch_running=False,
    experience=False,
    stream=False,
    trained_experience=False,

) -> List[PluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param minibatch: If True, will return a metric able to log
        the minibatch accuracy at training time.
    :param epoch: If True, will return a metric able to log
        the epoch accuracy at training time.
    :param epoch_running: If True, will return a metric able to log
        the running epoch accuracy at training time.
    :param experience: If True, will return a metric able to log
        the accuracy on each evaluation experience.
    :param stream: If True, will return a metric able to log
        the accuracy averaged over the entire evaluation stream of experiences.
    :param trained_experience: If True, will return a metric able to log
        the average evaluation accuracy only for experiences that the
        model has been trained on

    :return: A list of plugin metrics.
    """

    metrics = []
    if minibatch:
        metrics.append(MinibatchAccuracy(accuracy_fun))

    if epoch:
        metrics.append(EpochAccuracy(accuracy_fun))

    if epoch_running:
        metrics.append(RunningEpochAccuracy(accuracy_fun))

    if experience:
        metrics.append(ExperienceAccuracy(accuracy_fun))

    if stream:
        metrics.append(StreamAccuracy(accuracy_fun))

    if trained_experience:
        metrics.append(TrainedExperienceAccuracy(accuracy_fun))

    return metrics


__all__ = [
    "MyAccuracyPluginMetric",
    "MinibatchAccuracy",
    "EpochAccuracy",
    "RunningEpochAccuracy",
    "ExperienceAccuracy",
    "StreamAccuracy",
    "TrainedExperienceAccuracy",
    "accuracy_metrics",
]
