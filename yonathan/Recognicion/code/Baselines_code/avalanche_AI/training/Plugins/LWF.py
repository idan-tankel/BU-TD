"""
LWF plugin.
Penalties the current and previous
Output distributions KL divergence.
"""
import argparse
import sys
from typing import Union

import torch
import torch.nn as nn
from torch import Tensor

from Baselines_code.baselines_utils import construct_flag
from training.Data.Structs import inputs_to_struct, outs_to_struct
from training.Data.Data_params import RegType
from Baselines_code.avalanche_AI.training.Plugins.plugins_base import Base_plugin

sys.path.append(r'/')

KLoss = torch.nn.KLDivLoss(reduction='none')


class LwF(Base_plugin):
    """
    Learning without Forgetting plugin.
    LwF uses distillation to regularize the current loss with soft targets
    taken from a previous version of the model.
    This plugin does not use task identities.
    When used with multi-headed models, all heads are distilled.
    """

    def __init__(self, opts: argparse, prev_model: Union[dict, None] = None):
        """
        Args:
            opts: The model opts.
            prev_model: The prev_model if exists.
        """

        super(LwF, self).__init__(opts=opts, prev_checkpoint=prev_model, reg_type=RegType.LWF)
        self.temperature = opts.temperature_LWF  # The temperature.
        #     self.trained_tasks.append([0, (0, 1)])
        # self.trained_tasks.append([0, (-2, 0)])
        self.trained_tasks = [[0, (1, 0)]]
        self.trained_tasks.append([0, (0, 1)])
    #    self.trained_tasks.append([0, (-1, -1)])
        self.prev_tasks = dict()
        if prev_model is not None:
            self.num_exp = 1  # Number of trained experiences is set to 1.
            # Creating the desired flags for each trained task.
            for i, task in enumerate(self.trained_tasks):
                (task_id, direction_id) = task  # The task ,direction id.
                flag = construct_flag(opts, task_id, direction_id)  # Construct the flag.
                self.prev_tasks[i] = ((task_id, direction_id), flag)  # Construct the dictionary.

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

    def penalty(self, model: nn.Module, x: inputs_to_struct, **kwargs) -> Tensor.float:
        """
        Compute weighted distillation loss.
        Args:
            model: The model.
            x: The input.

        Returns: The penalty.

        """
        dist_loss = 0
        old_flag = x.flags  # Store the old flag.
        for _, New_flag in self.prev_tasks.values():
            x.flags = New_flag  # Set the new flag to activate the appropriate task-head.
            y_prev = self.prev_model.forward_and_out_to_struct(x)  # The previous distribution.
            y_curr = model.forward_and_out_to_struct(x)  # The current distribution.
            dist_loss += self._distillation_loss(y_curr, y_prev, x)  # The KL div loss.
        x.flags = old_flag  # return to the original flag.
        return self.reg_factor * dist_loss
