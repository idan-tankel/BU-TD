import sys
sys.path.append(r'/')
import copy
from avalanche.training.utils import get_last_fc_layer, freeze_everything
import torch
from avalanche.training.plugins import LFLPlugin
from typing import Union
import torch.nn as nn
from avalanche.training.templates.supervised import SupervisedTemplate

class MyLFLPlugin(LFLPlugin):
    """Less-Forgetful Learning (LFL) Plugin.

    LFL satisfies two properties to mitigate catastrophic forgetting.
    1) To keep the decision boundaries unchanged
    2) The feature space should not change much on target(new) data
    LFL uses euclidean loss between features from current and previous version
    of model as regularization to maintain the feature space and avoid
    catastrophic forgetting.
    Refer paper https://arxiv.org/pdf/1607.00122.pdf for more details
    This plugin does not use task identities.
    """

    def __init__(self, lambda_e:float, prev_model:Union[nn.Module, None] = None):
        """
        Create LFL plugin, if prev_model is not None, we copy its parameters.
        Args:
            lambda_e: The regularization factor.
            prev_model: The prev model.
        """
        super().__init__(lambda_e = lambda_e)
        if prev_model!= None:
          self.prev_model = copy.deepcopy(prev_model)
          freeze_everything(self.prev_model)

    def _euclidean_loss(self, features:torch, prev_features:torch)->float:
        """
        Compute euclidean loss.
        Args:
            features: The current model features.
            prev_features: The previous model features.

        Returns: The MSE loss.

        """
        return torch.nn.functional.mse_loss(features, prev_features)

    def penalty(self, x:list[torch], model:nn.Module, lambda_e:float):
        """
        Compute weighted euclidean loss
        Args:
            x: The input to the model.
            model: The current model.
            lambda_e: The regulation factor.

        Returns: The weighted MSE loss.

        """
        if self.prev_model is None:
            return 0
        else:
            features, prev_features = self.compute_features(model, x)
            dist_loss = self._euclidean_loss(features, prev_features)
            return lambda_e * dist_loss

    def compute_features(self, model:nn.Module, x:list[torch]):
        """
        Compute features from prev model and current model
        Args:
            model: The current model.
            x: The input x.

        Returns: The old, new features.

        """
        model.eval()
        self.prev_model.eval()
        features = model.forward_and_out_to_struct(x).features
        prev_features = self.prev_model.forward_and_out_to_struct(x).features
        return features, prev_features

    def before_backward(self, strategy:SupervisedTemplate, **kwargs)->None:
        """
        Add euclidean loss between prev and current features as penalty
        Args:
            strategy: The LFL strategy.

        """
        lambda_e = ( self.lambda_e[strategy.clock.train_exp_counter] if isinstance(self.lambda_e, (list, tuple)) else self.lambda_e)
        penalty = self.penalty(strategy.mb_x, strategy.model, lambda_e)
      #  print(penalty)
        strategy.loss += penalty

    def after_training_exp(self, strategy:SupervisedTemplate, **kwargs)->None:
        """
        Save a copy of the model after each experience
        and freeze the prev model and freeze the last layer of current model
        """
        if strategy.EpochClock.train_exp_epochs == strategy.EpochClock.max_epochs:
            print("freeze")
            self.prev_model = copy.deepcopy(strategy.model)
            freeze_everything(self.prev_model)
            last_fc_name, last_fc = get_last_fc_layer(strategy.model)
            for param in last_fc.parameters():
                param.requires_grad = False

    def before_training(self, strategy, **kwargs):
        pass