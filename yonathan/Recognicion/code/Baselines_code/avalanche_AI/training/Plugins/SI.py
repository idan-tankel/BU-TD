import argparse
import copy
import os
import sys

import torch
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.templates.supervised import SupervisedTemplate

sys.path.append(r'/')


class SI(SupervisedPlugin):
    """
    EWC plugin.
    Stores for each parameter its importance.
    """

    def __init__(self, parser: argparse, prev_model=None, eps=1e-7):
        """
        Args:
            parser: The model parser.
            mode: The training mode.
            keep_importance_data: Whether to keep the importance Data_Creation.
            prev_model: A pretrained model
            prev_data: The old dataset.
        """
        super().__init__()
        self.copy_model = copy.deepcopy(prev_model)  # The previous model.
        self.parser = parser  # The model opts.
        self.num_exp = 0  # The number of exp trained so far.
        self.inputs_to_struct = parser.inputs_to_struct  # The inputs to struct method.
        self.outs_to_struct = parser.outs_to_struct  # The outputs to struct method
        self.si_lambda = parser.si_lambda
        self.eps = eps
        self.device = parser.device
        self.w = {n: torch.zeros(p.shape).to(self.device) for n, p in
                  self.copy_model.feature_extractor.named_parameters() if p.requires_grad}
        # Store current parameters as the initial parameters before first task starts
        self.older_params = {n: p.clone().detach().to(self.device) for n, p in
                             self.copy_model.feature_extractor.named_parameters()
                             if p.requires_grad}
        # Store importance weights matrices
        self.importances = {n: torch.zeros(p.shape).to(self.device) for n, p in
                            self.copy_model.feature_extractor.named_parameters()
                            if p.requires_grad}
        self.unreg_grads = {n: torch.zeros(p.shape) for n, p in prev_model.feature_extractor.named_parameters()
                            if p.requires_grad}
        self.curr_feat_ext = {n: p.clone().detach() for n, p in prev_model.feature_extractor.named_parameters() if
                              p.requires_grad}

    def before_backward(self, strategy: SupervisedTemplate, **kwargs) -> None:
        """
        Compute EWC penalty and add it to the loss.
        Args:
            strategy: The strategy.

        """
        self.curr_feat_ext = {n: p.clone().detach() for n, p in strategy.model.feature_extractor.named_parameters() if
                              p.requires_grad}
        loss = strategy.loss
        loss.backward(retain_graph=True)
        self.unreg_grads = {n: p.grad.clone().detach() for n, p in strategy.model.feature_extractor.named_parameters()
                            if p.grad is not None}
        if self.num_exp > 0:
            # store gradients without regularization term
            # apply loss with path integral regularization
            strategy.loss += self.reg_penalty(strategy=strategy)
        strategy.optimizer.zero_grad()

    def after_update(self, strategy, **kwargs):
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
            loss_reg += torch.sum(self.importances[n] * (p - self.older_params[n]).pow(2))
            # Current cross-entropy loss -- with exemplars use all heads

        return self.si_lambda * loss_reg

    def after_training_exp(self, strategy: SupervisedTemplate, **kwargs) -> None:
        """
        Compute importance of parameters after each experience.
        Args:
            strategy: The strategy.

        """
        with torch.no_grad():
            curr_params = {n: p for n, p in strategy.model.feature_extractor.named_parameters() if p.requires_grad}
            for n, p in self.importances.items():
                p += self.w[n] / ((curr_params[n] - self.older_params[n]) ** 2 + self.eps)
                self.w[n].zero_()

        path = os.path.join(strategy.Model_folder, strategy.model.__class__.__name__ + f'_latest_direction'
                                                                                       f'={strategy.direction_id}.pt')
        state = torch.load(path)
        state['SI_importances'] = self.importances
        self.older_params = {n: p.clone().detach() for n, p in strategy.model.feature_extractor.named_parameters() if
                             p.requires_grad}
