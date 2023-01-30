"""
SI plugin.
Similar to EWC, computes
coefficients and penalties by
quadratic loss.
"""
import argparse
import copy
import sys
from typing import Optional

import torch
import torch.nn as nn
from Baselines_code.avalanche_AI.training.Plugins.plugins_base import Base_plugin
from Baselines_code.baselines_utils import compute_quadratic_loss
from avalanche.training.templates.supervised import SupervisedTemplate
from training.Data.Data_params import RegType
from training.Data.Structs import inputs_to_struct

sys.path.append(r'/')


class SI(Base_plugin):
    """
    EWC plugin.
    Stores for each parameter its importance.
    """

    def __init__(self, opts: argparse, prev_checkpoint: Optional[dict], eps=1e-7):
        """
        Args:
            opts: The model opts.
            prev_checkpoint: A pretrained model
            eps: The epsilon needed for non-zero devising.
        """
        super(SI, self).__init__(opts=opts, prev_checkpoint=prev_checkpoint, reg_type=RegType.SI)
        self.eps = eps

        self.w = {n: torch.zeros(p.shape).to(self.device) for n, p in
                  self.prev_model.feature_extractor().named_parameters() if p.requires_grad}
        # Store current parameters as the initial parameters before first task starts
        self.older_params = {n: p.clone().detach().to(self.device) for n, p in
                             self.prev_model.feature_extractor().named_parameters()
                             if p.requires_grad}
        # Store importance weights matrices
        self.importances = {n: torch.zeros(p.shape).to(self.device) for n, p in
                            self.prev_model.feature_extractor().named_parameters()
                            if p.requires_grad}
        self.grads = {n: torch.zeros(p.shape) for n, p in self.prev_model.feature_extractor().named_parameters()
                      if p.requires_grad}
        self.curr_feat_ext = {n: p.clone().detach() for n, p in self.prev_model.feature_extractor().named_parameters()
                              if
                              p.requires_grad}
        self.num_exp = 1
        if prev_checkpoint is not None:
            try:
                self.importances = prev_checkpoint['SI_state_dict']['regulizer_state_dict']
                print("success")
            except KeyError or TypeError:
                print("Failure")

    def before_backward(self, strategy: SupervisedTemplate, **kwargs) -> None:
        """
        Compute EWC penalty and add it to the loss.
        Args:
            strategy: The strategy.

        """
        self.curr_feat_ext = {n: p.clone().detach() for n, p in strategy.model.feature_extractor().named_parameters() if
                              p.requires_grad}
        loss = strategy.loss
        loss.backward(retain_graph=True)
        self.grads = {n: p.grad.clone().detach() for n, p in strategy.model.feature_extractor().named_parameters()
                      if p.grad is not None}
        if self.num_exp > 0:
            # store gradients without regularization term
            # apply loss with path integral regularization
            super(SI, self).before_backward(strategy=strategy)
        strategy.optimizer.zero_grad()

    def after_update(self, strategy, **kwargs):
        """

        Args:
            strategy:
            **kwargs:
        """
        # Eq. 3: accumulate w, compute the path integral -- "In practice, we can approximate w online as the running
        #  sum of the product of the gradient with the parameter update".
        with torch.no_grad():
            for n, p in strategy.model.feature_extractor().named_parameters():
                if n in self.grads.keys():
                    # w[n] >=0, but minus for loss decrease
                    self.w[n] -= self.grads[n] * (p.detach() - self.curr_feat_ext[n])

    def penalty(self, model: nn.Module, mb_x: inputs_to_struct, **kwargs):
        """Returns the loss value"""
        return compute_quadratic_loss(model, self.prev_model, importance=self.importances,
                                      device=self.device)

    def state_dict(self, strategy: SupervisedTemplate):
        """

        Args:
            strategy:

        Returns:

        """
        new_importances = copy.deepcopy(self.importances)
        new_importances = self.compute_new_importances(new_importances=new_importances, zero=False, strategy=strategy)
        return new_importances

    def compute_new_importances(self, new_importances, strategy, zero):
        """

        Args:
            new_importances:
            strategy:
            zero:

        Returns:

        """
        with torch.no_grad():
            curr_params = {n: p for n, p in strategy.model.feature_extractor().named_parameters() if p.requires_grad}
            for n, p in self.importances.items():
                new_importances[n] = self.importances[n] + self.w[n] / ((curr_params[n] - self.older_params[n]) ** 2 +
                                                                        self.eps)
                if zero:
                    self.w[n].zero_()
            return new_importances

    def after_training_exp(self, strategy: SupervisedTemplate, **kwargs) -> None:
        """
        Compute importance of parameters after each experience.
        Args:
            strategy: The strategy.

        """
        self.importances = self.compute_new_importances(new_importances=self.importances, zero=True, strategy=strategy)
        self.older_params = {n: p.clone().detach() for n, p in strategy.model.feature_extractor().named_parameters() if
                             p.requires_grad}
