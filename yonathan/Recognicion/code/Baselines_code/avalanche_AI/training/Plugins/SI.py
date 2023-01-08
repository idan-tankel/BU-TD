import copy
import sys
from torch.utils.data import Dataset
from training.Utils import preprocess
from torch.utils.data import DataLoader
import torch
from avalanche.models.utils import avalanche_forward
from avalanche.training.utils import copy_params_dict, zerolike_params_dict
from avalanche.training.plugins import EWCPlugin
import argparse
import torch.nn as nn
from typing import Callable
from torch.optim import optimizer
from Baselines_code.baselines_utils import set_model
from avalanche.training.templates.supervised import SupervisedTemplate

sys.path.append(r'/')

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin


class SI(SupervisedPlugin):
    """
    EWC plugin.
    Stores for each parameter its importance.
    """

    def __init__(self, parser: argparse, prev_model = None,damping = 0.1):
        """
        Args:
            parser: The model parser.
            mode: The training mode.
            keep_importance_data: Whether to keep the importance Data_Creation.
            prev_model: A pretrained model
            prev_data: The old dataset.
        """
        super().__init__()
        self.prev_model = copy.deepcopy(prev_model)  # The previous model.
        self.parser = parser  # The model opts.
        self.num_exp = 0  # The number of exp trained so far.
        self.inputs_to_struct = parser.inputs_to_struct  # The inputs to struct method.
        self.outs_to_struct = parser.outs_to_struct  # The outputs to struct method
        self.si_lambda = parser.si_lambda
        self.damping = damping
        self.clipgrad = 10000
        self.device = parser.device
        self.w = {n: torch.zeros(p.shape).to(self.device) for n, p in self.prev_model.feature_extractor.named_parameters() if p.requires_grad}
        # Store current parameters as the initial parameters before first task starts
        self.older_params = {n: p.clone().detach().to(self.device) for n, p in self.prev_model.feature_extractor.named_parameters()
                             if p.requires_grad}
        # Store importance weights matrices
        self.importance = {n: torch.zeros(p.shape).to(self.device) for n, p in self.prev_model.feature_extractor.named_parameters()
                           if p.requires_grad}

        # Supporting pretrained model.
        if prev_model is not None:
            # Update importance and old params to begin with EWC training.
            print('Done computing Importances')
            #     self.saved_params[0] = dict(copy_params_dict(prev_model.feature_extractor))  # Copy the old parameters.
            self.num_exp = 0  #

    def after_training_exp(self, strategy: SupervisedTemplate) -> None:
        """
        Compute importance of parameters after each experience.
        Args:
            strategy: The strategy.

        """
        with torch.no_grad():
            curr_params = {n: p for n, p in strategy.model.feature_extractor.named_parameters() if p.requires_grad}
            for n, p in self.importance.items():
                p += self.w[n] / ((curr_params[n] - self.older_params[n]) ** 2 + self.damping)
                self.w[n].zero_()
        self.older_params = {n: p.clone().detach() for n, p in strategy.model.feature_extractor.named_parameters() if p.requires_grad}

    def before_backward(self, strategy: SupervisedTemplate) -> None:
        """
        Compute EWC penalty and add it to the loss.
        Args:
            strategy: The strategy.

        """
        self.curr_feat_ext = {n: p.clone().detach() for n, p in strategy.model.feature_extractor.named_parameters() if p.requires_grad}
        loss = strategy.criterion()
        loss.backward(retain_graph=True)
        self.unreg_grads = {n: p.grad.clone().detach() for n, p in strategy.model.feature_extractor.named_parameters()
                       if p.grad is not None}
        # store gradients without regularization term
        # apply loss with path integral regularization
        if self.num_exp > 0:
            strategy.loss += self.reg_penalty(strategy=strategy)
        strategy.optimizer.zero_grad()
        print(self.reg_penalty(strategy=strategy))

    def after_update(self, strategy):
        torch.nn.utils.clip_grad_norm_(strategy.model.parameters(), self.clipgrad)
        # Eq. 3: accumulate w, compute the path integral -- "In practice, we can approximate w online as the running
        #  sum of the product of the gradient with the parameter update".
        with torch.no_grad():
            for n, p in strategy.model.feature_extractor.named_parameters():
                if n in self.unreg_grads.keys():
                    # w[n] >=0, but minus for loss decrease
                    self.w[n] -= self.unreg_grads[n] * (p.detach() - self.curr_feat_ext[n])

    def reg_penalty(self, strategy):
        """Returns the loss value"""
        loss_reg = 0
        # Eq. 4: quadratic surrogate loss
        for n, p in strategy.model.feature_extractor.named_parameters():
          #  print(self.w[n])
            loss_reg += torch.sum(self.importance[n] * (p - self.older_params[n]).pow(2))
            # Current cross-entropy loss -- with exemplars use all heads

        return self.si_lambda * loss_reg

