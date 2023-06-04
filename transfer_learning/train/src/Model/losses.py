"""
Here we define different loss functions needed for ourselves and for LwF.
"""
import copy

import torch.nn as nn
import torch

import argparse

import torch.nn.functional as F

from ..data.Enums import TrainingFlag


class LwFLoss(nn.Module):
    def __init__(self, reg_factor, old_model):
        super(LwFLoss, self).__init__()
        self.old_model = copy.deepcopy(old_model)
        self.reg_factor = reg_factor

    def forward(self, inputs: list, model: nn.Module, mode: str):
        """
        Compute forward pass.
        Args:
            mode: The pass mode.
            inputs: The inputs.
            model: The Model.

        Returns: The outs, loss

        """
        self.old_model.eval()
        images, labels, flags = inputs
        feature_outs = model(images, flags)
        if mode == 'train':
            new_feature = nn.Dropout(0.2)(feature_outs)
        else:
            new_feature = feature_outs
        outs = model.classifier(new_feature, flags)
        class_loss = nn.CrossEntropyLoss()(outs, labels)
        reg_loss = self.lwf_loss(inputs, feature_outs)
        return outs, class_loss + self.reg_factor * reg_loss

    def lwf_loss(self, inputs, feature_outs):
        """
        LwF loss.
        Args:
            inputs: The inputs.
            feature_outs: The feature output.

        Returns: The distillation loss.

        """
        images, _, flags = inputs
        cur_out = self.old_model.classifier(feature_outs, [0])
        prev_out = self.old_model.classifier(self.old_model(images, flags), [0])
        cur_out_log_softmax = torch.log_softmax(cur_out, dim=1)  # Compute the log-probabilities.
        prev_out_softmax = torch.softmax(prev_out, dim=1)  # Compute the probabilities.
        dist_loss = F.kl_div(cur_out_log_softmax, prev_out_softmax)  # Compute the loss.
        return dist_loss


class RegLoss(nn.Module):
    def __init__(self, reg_factor, learned_params):
        super(RegLoss, self).__init__()
        self.reg = reg_factor
        self.learned_params = learned_params
        self.old_params = copy.deepcopy(learned_params)

    def forward(self, inputs: list, model: nn.Module, mode: str):
        """
        Forward of regularization loss.
        Args:
            inputs: The inputs.
            model: The Model.
            mode: The mode.

        Returns: The outs, loss

        """
        img, labels, flags = inputs
        outs = model(img, flags)
        if mode == 'train':
            outs = nn.Dropout(0.2)(outs)
        outs = model.classifier(outs, flags)
        class_loss = nn.CrossEntropyLoss()(outs, labels)
        class_loss += self.reg * sum([torch.norm(param) ** 2 for param in self.learned_params])
        return outs, class_loss


def Get_loss_fun(training_flag: TrainingFlag, opts: argparse, model: nn.Module, learned_params: list):
    """
    Get the loss function.
    Args:
        training_flag: The training flag.
        opts: The Model opts.
        model: The Model.
        learned_params: The learned params.

    Returns: The desired loss function.

    """
    if training_flag is TrainingFlag.LWF:
        try:
            reg_fac = opts.data_set_obj['reg']
        except AttributeError:
            reg_fac = .0
        loss_fun = LwFLoss(reg_factor=reg_fac, old_model=model)
    else:
        loss_fun = RegLoss(opts.data_set_obj['reg'], learned_params)

    return loss_fun
