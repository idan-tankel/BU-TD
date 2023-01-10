import argparse
import copy
import sys

import torch
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.templates.supervised import SupervisedTemplate
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from Baselines_code.baselines_utils import compute_quadratic_loss, compute_fisher_information_matrix

sys.path.append(r'/')


class MyIMM_Mode(SupervisedPlugin):
    """
    EWC plugin.
    Stores for each parameter its importance.
    """

    def __init__(self, parser: argparse, prev_model=None, old_dataset: Dataset = None, load_from=None):
        """
        Args:
            parser: The model parser.
            mode: The training mode.
            keep_importance_data: Whether to keep the importance Data_Creation.
            prev_model: A pretrained model
            prev_data: The old dataset.
        """
        super().__init__()
        self.parser = parser  # The model opts.
        self.num_exp = 0  # The number of exp trained so far.
        self.inputs_to_struct = parser.inputs_to_struct  # The inputs to struct method.
        self.outs_to_struct = parser.outs_to_struct  # The outputs to struct method
        self.imm_mode_lambda = parser.IMM_Mode_lambda  # The regularization factor.
        self.criterion = parser.criterion  # The loss needed for fisher computation.
        self.device = parser.device
        self.bs = parser.bs
        self.load_from = load_from
        # Supporting pretrained model.
        if prev_model is not None:
            self.prev_model = copy.deepcopy(prev_model)  # The previous model.
            self.num_exp = 1  #
            dataloader = DataLoader(old_dataset, batch_size=self.bs)  # The dataloader.
            model_meta_data = torch.load(load_from)
            try:
                self.importances = model_meta_data['EWC_importances']
            except KeyError:
                self.fisher = compute_fisher_information_matrix(self.parser, prev_model, self.criterion, dataloader,
                                                                self.parser.device, self.parser.bs)
                model_meta_data['EWC_importances'] = self.importances
                torch.save(model_meta_data, load_from)
                print("Computed EWC importances once for all!")
            self.importances = {name: 1 for name, _ in prev_model.feature_extractor.named_parameters()}
            print('Done computing Importances')

    def after_training_exp(self, strategy: SupervisedTemplate) -> None:
        """
        Compute importance of parameters after each experience.
        Args:
            strategy: The strategy.

        """
        fisher_new = compute_fisher_information_matrix(self.parser, strategy.model, strategy._criterion,
                                                       strategy.dataloader, self.device, self.bs)
        for (n, curr_param), (_, prev_param) in zip(strategy.model.feature_extractor.named_parameters(),
                                                    self.prev_model.feature_extractor.named_parameters()):
            curr_param = fisher_new[n] * curr_param + self.fisher[n] * prev_param
            self.fisher[n] += fisher_new[n]
            curr_param /= (self.fisher[n] == 0).float() + self.fisher[n]

        torch.save(strategy.model.state_dict(), strategy.checkpoint.dir_path)

    def before_backward(self, strategy: SupervisedTemplate) -> None:
        """
        Compute EWC penalty and add it to the loss.
        Args:
            strategy: The strategy.

        """
        exp_counter = strategy.clock.train_exp_counter + self.num_exp
        if exp_counter == 0 or self.imm_mode_lambda == 0.0:
            return

        penalty = compute_quadratic_loss(strategy.model, self.prev_model, importance=self.importances,
                                         device=strategy.device)
        print(penalty)
        strategy.loss += self.imm_mode_lambda * penalty
