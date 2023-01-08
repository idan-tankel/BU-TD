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
class MyIMM_Mean_Plugin(SupervisedPlugin):
    """
    EWC plugin.
    Stores for each parameter its importance.
    """

    def __init__(self, parser: argparse,  prev_model=None):
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
        self.imm_mean_lambda = parser.imm_mean_lambda
        # Supporting pretrained model.
        if prev_model is not None:
            # Update importance and old params to begin with EWC training.
            print('Done computing Importances')
       #     self.saved_params[0] = dict(copy_params_dict(prev_model.feature_extractor))  # Copy the old parameters.
            self.num_exp = 1  #

    def after_training_exp(self, strategy: SupervisedTemplate) -> None:
        """
        Compute importance of parameters after each experience.
        Args:
            strategy: The strategy.

        """
        exp_counter = strategy.clock.train_exp_counter + self.num_exp
        current_model_state = strategy.model.feature_extractor.state_dict()
        prev_model_state = self.prev_model.feature_extractor.state_dict()
        for name, param in strategy.model.feature_extractor.named_parameters():
            current_model_state[name] = (current_model_state[name] + prev_model_state[name] * exp_counter) / (exp_counter + 1)
        set_model(strategy.model.feature_extractor, current_model_state)
        model_path_curr = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/Emnist/Baselines/'
        torch.save(strategy.model.state_dict(), model_path_curr)
        

    def before_backward(self, strategy: SupervisedTemplate) -> None:
        """
        Compute EWC penalty and add it to the loss.
        Args:
            strategy: The strategy.

        """
        exp_counter = strategy.clock.train_exp_counter + self.num_exp
        if exp_counter == 0 or self.imm_mean_lambda == 0.0:
            return
        penalty = torch.tensor(0).float().to(strategy.device)

        for (_, param), (_, param_old) in zip(strategy.model.feature_extractor.named_parameters(), self.prev_model.feature_extractor.named_parameters()):
            penalty += torch.sum((param - param_old).pow(2)) 
        # Update the new loss.
        strategy.loss += self.imm_mean_lambda * penalty