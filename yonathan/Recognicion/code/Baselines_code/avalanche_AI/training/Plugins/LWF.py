import sys
sys.path.append(r'/')
import copy
import argparse
import torch
from avalanche.training.plugins import LwFPlugin
from typing import Union
import torch.nn as nn

class MyLwFPlugin(LwFPlugin):
    """
    A Learning without Forgetting plugin.
    LwF uses distillation to regularize the current loss with soft targets
    taken from a previous version of the model.
    This plugin does not use task identities.
    When used with multi-headed models, all heads are distilled.
    """

    def __init__(self, parser:argparse, prev_model:Union[nn.Module, None] = None):
        """
        Args:
            parser: The model parser.
            prev_model: The prev_model if exists.
        """

        super().__init__()

        self.alpha = parser.alpha_LWF
        self.temperature =parser.temperature_LWF
        self.prev_model = copy.deepcopy(prev_model) if prev_model != None else None
        self.prev_classes = {"0": set()}

    def _distillation_loss(self, cur_out:torch, prev_out:torch, x:list[torch]):
        """
        Compute distillation loss between output of the current model and
        and output of the previous (saved) model.
        Args:
            cur_out: The current output.
            prev_out: The previous output.
            x: The input.

        Returns: The distillation loss.

        """
        loss_weight = x.label_existence.unsqueeze(dim = 2) # Getting the loss weight.
        cur_out = torch.transpose(cur_out.classifier, 2, 1)
        prev_out = torch.transpose(prev_out.classifier, 2, 1)
        cur_out_softmax = torch.log_softmax(cur_out / self.temperature, dim = 2 )
        prev_out_softmax = torch.softmax(prev_out / self.temperature, dim = 2)
        dist_loss = - cur_out_softmax * prev_out_softmax
        dist_loss = dist_loss * loss_weight
        dist_loss = dist_loss.sum() / loss_weight.size(0)
        return dist_loss

    def penalty(self, model:nn.Module, x:list[torch], alpha:float):
        """
        Compute weighted distillation loss.
        Args:
            model: The model.
            x: The input.
            alpha: The regularization factor.

        Returns: The penalty.

        """
        # TODO - CHANGE TO THE TASK IDENTITIES.
        if self.prev_model is None:
            return 0
        else:

            y_prev = {"0": self.prev_model.forward_and_out_to_struct(x, head = 0)  }
            y_curr = {"0": model.forward_and_out_to_struct(x, head = 0)  }
            dist_loss = 0
            for task_id in y_prev.keys():
                # compute kd only for previous heads.
                if task_id in self.prev_classes:
                    yp = y_prev[task_id]
                    yc = y_curr[task_id]
                    input_strcut = model.inputs_to_struct(x)
                    dist_loss += self._distillation_loss(yc, yp, input_strcut)
            return alpha * dist_loss

    def before_backward(self, strategy, **kwargs):
        """
        Summing all losses together.
        Args:
            strategy: The strategy.
            **kwargs: 

        """
        alpha = ( self.alpha[strategy.clock.train_exp_counter]   if isinstance(self.alpha, (list, tuple)) else self.alpha  )
        penalty = self.penalty(strategy.model, strategy.mb_x, alpha )
        print(penalty)
        strategy.loss += penalty

    def after_training_exp(self, strategy, **kwargs):
        """
        Save a copy of the model after each experience and
        update self.prev_classes to include the newly learned classes.
        # TODO - GET RID OF MY CLOCK.
        """
        if strategy.EpochClock.train_exp_epochs == strategy.EpochClock.max_epochs:
            print("copy")
            self.prev_model = copy.deepcopy(strategy.model)
        # For class incremental problems only.
        task_ids = strategy.experience.dataset.task_set
        for task_id in task_ids:
            task_data = strategy.experience.dataset.task_set[task_id]
            pc = set(task_data.targets)

            if task_id not in self.prev_classes:
                self.prev_classes[str(task_id)] = pc
            else:
                self.prev_classes[str(task_id)] = self.prev_classes[task_id].union(pc)