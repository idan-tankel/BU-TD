import argparse
import copy
import sys
from typing import Union

import torch
import torch.nn as nn
from avalanche.training.plugins import LwFPlugin
from avalanche.training.templates.supervised import SupervisedTemplate as Regularization_strategy
from torch import Tensor

from Baselines_code.baselines_utils import construct_flag
from training.Data.Structs import inputs_to_struct, outs_to_struct

sys.path.append(r'/')

KLoss = torch.nn.KLDivLoss(reduction='none')


class LwF(LwFPlugin):
    """
    Learning without Forgetting plugin.
    LwF uses distillation to regularize the current loss with soft targets
    taken from a previous version of the model.
    This plugin does not use task identities.
    When used with multi-headed models, all heads are distilled.
    """

    def __init__(self, opts: argparse, prev_model: Union[nn.Module, None] = None):
        """
        Args:
            opts: The model parser.
            prev_model: The prev_model if exists.
        """

        super().__init__()
        self.parser = opts  # The parser.
        self.lamda = opts.LWF_lambda  # The LWF lambda.
        self.temperature = opts.temperature_LWF  # The temperature.
        self.num_exps = 0  # No exps trained before.
        self.inputs_to_struct = opts.inputs_to_struct  # The inputs to struct.
        if prev_model is not None:
            self.prev_model = copy.deepcopy(prev_model)  # Copy the previous model.
            self.num_exps = 1  # Number of trained experiences is set to 1.
            # Creating the desired flags for each trained task.
            for i, task in enumerate(prev_model.trained_tasks):
                (task_id, direction_id) = task  # The task ,direction id.
                flag = construct_flag(opts, task_id, direction_id)  # Construct the flag.
                self.prev_tasks = {i: ((task_id, direction_id), flag)}  # Construct the dictionary.

    def _distillation_loss(self, cur_out: outs_to_struct, prev_out: outs_to_struct,
                           x: inputs_to_struct) -> Tensor.float:
        """
        Compute distillation loss between output of the current model and
        output of the previous (saved) model.
        Args:
            cur_out: The current output.
            prev_out: The previous output.
            x: The input.

        Returns: The distillation loss.

        """

        loss_weight = x.label_existence.unsqueeze(dim=1)  # Expand to match the shape.
        cur_out_log_softmax = torch.log_softmax(cur_out.classifier / self.temperature,
                                                dim=1)  # Compute the log-probabilities.
        prev_out_softmax = torch.softmax(prev_out.classifier / self.temperature, dim=1)  # Compute the probabilities.
        dist_loss = KLoss(cur_out_log_softmax, prev_out_softmax)  # Compute the loss.
        dist_loss = (dist_loss * loss_weight).sum() / loss_weight.size(0)  # Count only existing characters.
        return dist_loss

    def penalty(self, model: nn.Module, x: inputs_to_struct) -> Tensor.float:
        """
        Compute weighted distillation loss.
        Args:
            model: The model.
            x: The input.

        Returns: The penalty.

        """

        if self.num_exps == 0:
            return 0.0
        else:
            dist_loss = 0
            old_flag = x.flags  # Store the old flag.
            for _, New_flag in self.prev_tasks.values():
                x.flags = New_flag  # Set the new flag to activate the appropriate task-head.
                y_prev = self.prev_model.forward_and_out_to_struct(x)  # The previous distribution.
                y_curr = model.forward_and_out_to_struct(x)  # The current distribution.
                dist_loss += self._distillation_loss(y_curr, y_prev, x)  # The KL div loss.
            x.flags = old_flag  # return to the original flag.
            return self.lamda * dist_loss

    def before_backward(self, strategy: Regularization_strategy, **kwargs) -> None:
        """
        Summing all losses together.
        Args:
            strategy: The strategy.
            **kwargs: Optional args.

        """
        # Compute the LWF penalty.
        penalty = self.penalty(strategy.model, strategy.mb_x)
        strategy.loss += penalty  # Add the penalty.

    def after_training_exp(self, strategy: Regularization_strategy, **kwargs) -> None:
        """
        Save a copy of the model after each experience.
        Args:
            strategy: The strategy.
            **kwargs:

        """
        print("Copy the model ")
        # Copy the current model.
        self.prev_model = copy.deepcopy(strategy.model)
