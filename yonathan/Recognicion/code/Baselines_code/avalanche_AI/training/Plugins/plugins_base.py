"""
Base class for all plugins.
Support backward, penalty, and model saving.
"""
import copy

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.templates.supervised import SupervisedTemplate as Regularization_strategy
from torch import Tensor
import torch.nn as nn
import argparse
from Baselines_code.baselines_utils import RegType
from training.Data.Structs import inputs_to_struct
from training.Modules.Create_Models import create_model
from training.Data.Get_dataset import get_dataset_for_spatial_relations


class Base_plugin(SupervisedPlugin):
    """
    Base Plugin.
    """

    def __init__(self, opts: argparse, reg_type: RegType, prev_checkpoint: nn.Module = None):
        super(SupervisedPlugin, self).__init__()
        self.prev_model = create_model(opts=opts)
        if prev_checkpoint is not None:
            self.prev_model.load_state_dict(state_dict=prev_checkpoint['model_state_dict'])
            self.num_exp = 1
        self.trained_tasks = [(0, (1, 0))]
        self.opts = opts  # The model opts.
        self.num_exp = 0  # The number of exp trained so far.
        self.inputs_to_struct = opts.inputs_to_struct  # The inputs to struct method.
        self.outs_to_struct = opts.outs_to_struct  # The outputs to struct method
        self.reg_factor = reg_type.class_to_reg_factor(opts)  # The regularization factor.
        self.device = opts.device  # The device

    def compute_convex_loss(self, ce_loss: Tensor.float, reg_loss: Tensor.float):
        """
        Args:
            ce_loss:
            reg_loss:

        Returns:

        """
        return self.reg_factor * reg_loss + (1 - self.reg_factor) * ce_loss

    def penalty(self, model: nn.Module, mb_x: inputs_to_struct, **kwargs):
        """
        The penalty.
        Args:
            model: The model.
            mb_x: The input.
            **kwargs:
        """
        raise NotImplementedError

    def before_backward(self, strategy: Regularization_strategy, **kwargs) -> None:
        """
        Summing all losses together.
        Args:
            strategy: The strategy.
            **kwargs: Optional args.

        """
        exp_counter = strategy.clock.train_exp_counter + self.num_exp
        if exp_counter == 0 or self.reg_factor == 0.0:
            return
        # Compute the strategy penalty.
        penalty = self.penalty(strategy.model, strategy.mb_x)
        strategy.loss = self.compute_convex_loss(strategy.loss, penalty)

    def state_dict(self, strategy: Regularization_strategy):
        """

        Returns:

        """
        return dict()

    def Get_old_dl(self):
        """

        Returns:

        """
        try:
            task = self.trained_tasks[-1]
            old_data = get_dataset_for_spatial_relations(self.opts, self.opts.Images_path, task[0], [task[1]])
            return old_data['train_dl']
        except IndexError:
            return None

    def after_training_exp(self, strategy: Regularization_strategy, *args, **kwargs):
        """
        Args:
            strategy:
            *args:
            **kwargs:
        """
        print("Copy the new model!")
        self.prev_model = copy.deepcopy(strategy.model)  # Copy the new version of the model.
