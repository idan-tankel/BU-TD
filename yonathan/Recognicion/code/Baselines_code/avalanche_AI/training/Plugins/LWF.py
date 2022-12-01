import sys
sys.path.append(r'/')
import copy
import argparse
import torch
from avalanche.training.plugins import LwFPlugin
from typing import Union
import torch.nn as nn
from training.Utils import tuple_direction_to_index
from avalanche.training.templates.supervised import SupervisedTemplate

class MyLwFPlugin(LwFPlugin):
    """
    Learning without Forgetting plugin.
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
        self.parser= parser
        self.lamda = parser.LWF_lambda
        self.temperature =parser.temperature_LWF
        self.num_exps = 0
        self.inputs_to_struct = parser.inputs_to_struct
        if prev_model is not None:
             self.prev_model = copy.deepcopy(prev_model)
             self.num_exps = 1
             for i, task in enumerate(prev_model.trained_tasks):
                self.prev_tasks = {i : task }

    #TODO - IT WORKS FOR RESNET ONLY, FOR BU-TD WE NEED TO CONSIDER SOLUTIONS.
    def construct_flag(self,flag, task_id, direction_id):
        B = flag.shape[0]
        direction_dir, _ = tuple_direction_to_index(self.parser.num_x_axis, self.parser.num_y_axis, direction_id, self.parser.ndirections, task_id)
        task_id = torch.tensor(task_id)
        New_task_flag = torch.nn.functional.one_hot(task_id, self.parser.ntasks)
        New_direction_flag = torch.nn.functional.one_hot(direction_dir, self.parser.ndirections)
        New_flag = torch.concat([New_direction_flag, New_task_flag], dim=0).float()
        return New_flag.expand(B,-1)

    def _distillation_loss(self, cur_out:torch, prev_out:torch, x:list[torch])->float:
        """
        Compute distillation loss between output of the current model and
        output of the previous (saved) model.
        Args:
            cur_out: The current output.
            prev_out: The previous output.
            x: The input.

        Returns: The distillation loss.

        """
        loss_weight = x.label_existence.unsqueeze(dim = 2) # Getting the loss weight.
        cur_out = torch.transpose(cur_out.classifier, 2, 1) # Get in the order of [B,CHAR,class].
        prev_out = torch.transpose(prev_out.classifier, 2, 1) # Get in the order of [B,CHAR,class].
        cur_out_softmax = torch.log_softmax(cur_out / self.temperature, dim = 2 ) # Compute the log-probabilities.
        prev_out_softmax = torch.softmax(prev_out / self.temperature, dim = 2)# Compute the probabilities.
        dist_loss = - cur_out_softmax * prev_out_softmax # Compute the loss.
        dist_loss = dist_loss * loss_weight # Count only existing characters.
        dist_loss = dist_loss.sum() / loss_weight.size(0) # Average the loss
        return dist_loss

    def penalty(self, model:nn.Module, x:list[torch], alpha:float)->float:
        """
        Compute weighted distillation loss.
        Args:
            model: The model.
            x: The input.
            alpha: The regularization factor.

        Returns: The penalty.

        """

        if self.num_exps == 0:
            return 0.0
        else:
            dist_loss = 0
           # input_struct = self.inputs_to_struct(x)
            # TODO - TOMORROW BUILD THIS FUNCTION.
         #   New_flag = construct_flag(input_struct.flag, old_task)
            # compute kd only for previous heads.
            for task_id, direction_id in self.prev_tasks.values():
                New_flag = self.construct_flag(x.flag, task_id, direction_id)
                x.flag = New_flag
                y_prev =  self.prev_model.forward_and_out_to_struct(x)
                y_curr =  model.forward_and_out_to_struct(x)
                dist_loss += self._distillation_loss(y_curr, y_prev, x)
            return alpha * dist_loss

    def before_backward(self, strategy:SupervisedTemplate, **kwargs):
        """
        Summing all losses together.
        Args:
            strategy: The strategy.
            **kwargs: 

        """
        alpha =  self.lamda
        penalty = self.penalty(strategy.model, strategy.mb_x, alpha )
        print(penalty)
        strategy.loss += penalty

    def after_training_exp(self, strategy:SupervisedTemplate, **kwargs):
        """
        Save a copy of the model after each experience and
        update self.prev_classes to include the newly learned classes.
        """
        print("Copy the model ")
        self.prev_model = copy.deepcopy(strategy.model)
        # For class incremental problems only.

