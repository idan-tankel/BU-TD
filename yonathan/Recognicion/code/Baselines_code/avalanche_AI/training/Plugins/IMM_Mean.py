import argparse
import copy
import os
import sys

import torch
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.templates.supervised import SupervisedTemplate as Regularization_strategy

from Baselines_code.baselines_utils import compute_quadratic_loss
from Baselines_code.baselines_utils import set_model

sys.path.append(r'/')


class IMM_Mean(SupervisedPlugin):
    """
    EWC plugin.
    Stores for each parameter its importance.
    """

    def __init__(self, parser: argparse, prev_model=None):
        """
        Args:
            parser: The model parser.
            prev_model: A pretrained model
        """
        super().__init__()
        self.parser = parser  # The model opts.
        self.num_exp = 0  # The number of exp trained so far.
        self.inputs_to_struct = parser.inputs_to_struct  # The inputs to struct method.
        self.outs_to_struct = parser.outs_to_struct  # The outputs to struct method
        self.imm_mean_lambda = parser.IMM_Mean_lambda  # The regularization factor.

        # Supporting pretrained model.
        if prev_model is not None:
            self.prev_model = copy.deepcopy(prev_model)  # The previous model
            self.num_exp = 1
            self.importances = {name: 1 for name, _ in prev_model.feature_extractor.named_parameters()}

    def after_training_exp(self, strategy: Regularization_strategy) -> None:
        """
        Compute importance of parameters after each experience.
        Args:
            strategy: The strategy.

        """
        exp_counter = strategy.clock.train_exp_counter + self.num_exp
        # Current model state.
        current_model_state = strategy.model.feature_extractor.state_dict()
        # Previous model state.
        prev_model_state = self.prev_model.feature_extractor.state_dict()
        # Update the current and previous model.
        for name, param in strategy.model.feature_extractor.named_parameters():
            current_model_state[name] = (current_model_state[name] + prev_model_state[name] * exp_counter) / (
                    exp_counter + 1)
        # Set the new weights into the model.
        set_model(strategy.model.feature_extractor, current_model_state)
        model_path_curr = os.path.join(strategy.checkpoint.dir_path, 'Merged_model')
        # Save the model.
        torch.save(strategy.model.state_dict(), model_path_curr)

    def before_backward(self, strategy: Regularization_strategy) -> None:
        """
        Compute EWC penalty and add it to the loss.
        Args:
            strategy: The strategy.

        """

        exp_counter = strategy.clock.train_exp_counter + self.num_exp
        if exp_counter == 0 or self.imm_mean_lambda == 0.0:
            return

        penalty = compute_quadratic_loss(strategy.model, self.prev_model, importance=self.importances,
                                         device=strategy.device)
        print(penalty)
        strategy.loss += self.imm_mean_lambda * penalty
