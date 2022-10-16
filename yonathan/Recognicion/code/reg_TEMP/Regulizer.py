import argparse
import os
import pickle

import torch
from supp.utils import cross_entropy
from torch.utils.data import DataLoader

from supp.data_functions import preprocess


class EWC():  # elastic weight consolidation.
    def __init__(self, opts: argparse, data_loader: DataLoader):
        """
        Implementation of the elastic weight consolidation.
        Args:
            opts: The model options.
            data_loader: The data - loader.
        """

        # Configurations.
        self.lamda = opts.lamda
        self.opts = opts
        self.model_old = opts.model_old
        self.model = opts.model
        self.old_params = {}
        self.fisher = {}
        self.dl = data_loader
        self.estimate_params()

    def estimate_fisher(self, data_loader: DataLoader) -> dict:
        """
        Given the training loader, we compute the fisher matrix.
        Args:
            data_loader: The training data-loader.

        Returns: A dictionary assigning for each parameter its fisher matrix of the same shape.

        """
        fisher = {}
        for n, p in self.model_old.named_parameters():
            fisher[n] = 0.0

        for inputs in data_loader:
            inputs = preprocess(inputs)
            outs = self.model_old(inputs)  # Compute the model output.
            loss = self.opts.loss_fun(self.model_old, inputs, outs)  # Compute the loss.
            #            opts.optimizer.zero_grad()  # Reset the optimizer.
            loss.backward()  # Do a backward pass.
            for n, p in self.model_old.named_parameters():
                if p.grad != None:
                    fisher[n] += p.grad ** 2 / len(data_loader)

        return fisher  # Return the loss and the output.

    def consolidate(self):
        """
        Saving the old parameters.
        Args:
            fisher: The fisher matrix dictionary.

        """
        for n, p in self.model_old.named_parameters():
            self.old_params[n] = p.clone()

    def loss(self) -> float:
        """
        Returns: The regularization loss of the EWC method.
        """

        losses = []
        fisher = self.fisher
        for n, p in self.model.named_parameters():
            # retrieve the consolidated mean and fisher information.
            old_param = self.old_params[n]
            losses.append((fisher[n] * (p - old_param) ** 2).sum())
        return self.lamda * sum(losses)

    def _is_on_cuda(self):
        return next(self.model.parameters()).is_cuda

    def estimate_params(self):
        """
        Estimation fisher & saving the old params.
        """
        path = os.path.join(self.opts.model_dir, 'Fisher_dict')
        if not os.path.exists(path):
            self.fisher = self.estimate_fisher(self.dl)
            path = open(path, 'wb')
            pickle.dump(self.fisher, path)
        else:
            with open(path, "rb") as new_data_file:
                self.fisher = pickle.load(new_data_file)
        self.consolidate()


class LFL():
    def __init__(self, opts):
        """
        # Less forgetting learning http://arxiv.org/abs/1607.0012.
        Args:
            opts: The model options.
        """
        self.lamda = opts.lamda
        self.opts = opts
        self.model = opts.model
        self.model_old = opts.model_old

    def loss(self, input: list[torch], outputs):
        """
        In this method we don't look at the heads, and just ensure the last layers(before the task-head) are similiar.
        As all tasks share the same body, we have single loss over all tasks.
        Args:
            input: The model input.
            outputs: The model outputs, a struct containing the output  before the readout and after the readout.

        Returns: The distillation loss.

        """
        layer_old = self.model_old(input)
        layer = outputs.before_read_out
        return self.lamda * torch.sum((layer - layer_old).pow(2))


class LWF():
    """ Class implementing the Learning Without Forgetting approach described in https://arxiv.org/abs/1606.09282 """

    def __init__(self, opts: argparse):
        """
        Args:
            opts: The model options.
            ntasks_solved_until_now: The number of tasks solved so far.
        """
        self.opts = opts
        self.ntasks_solved_until_now = opts.ntasks_solved_until_now
        self.model = opts.model
        self.model_old = opts.model_old
        self.T = 1

    def loss(self, inputs: list[torch], outputs):
        """
        Compute the learning without forgetting distillation loss.
        Args:
            inputs: The model inputs.
            outputs: The model outputs, a struct containing the output  before the readout and after the readout.

        Returns: The regularization loss.

        """
        loss_dist = 0.0
        outputs = outputs.after_read_out
        targets_old = self.model_old(inputs)
        for t_old in range(self.ntasks_solved_until_now):
            loss_dist += cross_entropy(outputs[t_old], targets_old[t_old], exp=1 / self.T)
        return self.lamda * loss_dist


class Regulizer():
    def __init__(self, opts, data_loader=None):
        reg_type = opts.reg_type
        if reg_type == 'EWC':
            self.reg = EWC(opts, data_loader)

        if reg_type == 'LFL':
            self.reg = LFL(opts)

        if reg_type == 'LWF':
            self.reg = LWF(opts)

    def loss_step(self, inputs, outputs):
        return self.reg.loss(inputs, outputs)


'''
opts = Model_Options_By_Flag_And_DsType(Flag=Flag.ZF, DsType=DsType.Omniglot)
parser = GetParser(opts=opts, language_idx=0,direction = 'left')
data_path = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/omniglot/samples_new/6_extended_0'
# Create the data for right.
[the_datasets, _ ,  test_dl, _ ,] = get_dataset_for_spatial_realtions(parser, data_path,lang_idx = 0, direction = 0)

ewc = EWC(40.0, parser, test_dl )

# TODO - CHANGE TO HAVE MODEL OUTS.
'''
