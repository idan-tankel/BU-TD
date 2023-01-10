import argparse
import copy
from typing import Callable
from typing import Union

import torch
import torch.nn as nn
from avalanche.models.utils import avalanche_forward
from avalanche.training.utils import copy_params_dict, zerolike_params_dict
from avalanche.training.utils import freeze_everything
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from Baselines_code.baselines_utils import construct_flag
from training.Data.Data_params import Flag
from training.Data.Data_params import RegType
from training.Data.Structs import inputs_to_struct, outs_to_struct
from training.Utils import preprocess

KLoss = torch.nn.KLDivLoss(reduction='none')
class Regulizers():
    def __init__(self, parser, prev_model, reg_type):
        self.parser = parser
        self.reg_type =  reg_type
        self.reg_factor = getattr(parser, reg_type.__str__() + '_lambda')
        self.inputs_to_struct = parser.inputs_to_struct  # The inputs to struct.
        self.outs_to_struct = parser.outs_to_struct  # The outputs to struct method
        self.model_flag = parser.model_flag
        self.device = parser.device
        self.train_mb_size = parser.train_mb_size
        self.prev_model =  copy.deepcopy(prev_model)
        freeze_everything(self.prev_model)

    def compute_reg_penalty(self, model, data_point):
        raise NotImplementedError

    def after_training_exp(self, model, dataset, optimizer):
        self.prev_model = copy.deepcopy(model)
        freeze_everything(self.prev_model)

class LFL(Regulizers):
    def _euclidean_loss(self, features: Tensor, prev_features: Tensor) -> Tensor.float:
        """
        Compute euclidean loss.
        Args:
            features: The current model features.
            prev_features: The previous model features.

        Returns: The MSE loss.

        """
        # The MSE loss.
        return torch.nn.functional.mse_loss(features, prev_features)

    def compute_features(self, model: nn.Module, x: list[torch]) -> tuple[torch, torch]:
        """
        Compute features from prev model and current model
        Args:
            model: The current model.
            x: The input to the models.

        Returns: The old, new features.

        """
        model.eval()  # Move to eval mode.
        self.prev_model.eval()  # Move to eval model.
        features = model.forward_and_out_to_struct(x).features  # New features.
        prev_features = self.prev_model.forward_and_out_to_struct(x).features  # Old features.
        return features, prev_features

    def compute_reg_penalty(self,  model: nn.Module, x: list[torch]) -> torch.float:
        """
        Compute weighted euclidean loss
        Args:
            x: The input to the model.
            model: The current model.
            lambda_e: The regulation factor.

        Returns: The weighted MSE loss.

        """
        # The previous, current features.
        features, prev_features = self.compute_features(model, x)
        # Compute distance loss.
        dist_loss = torch.nn.functional.mse_loss(features, prev_features)
        return self.reg_factor * dist_loss

class LWF(Regulizers):
    """
    Learning without Forgetting plugin.
    LwF uses distillation to regularize the current loss with soft targets
    taken from a previous version of the model.
    This plugin does not use task identities.
    When used with multi-headed models, all heads are distilled.
    """

    def __init__(self, parser: argparse, reg_type, prev_model: Union[nn.Module, None] = None):
        """
        Args:
            parser: The model parser.
            prev_model: The prev_model if exists.
        """

        super().__init__(parser=parser, prev_model=prev_model, reg_type=reg_type)
        self.temperature = parser.temperature_LWF  # The temperature.
        # Creating the desired flags for each trained task.
        for i, task in enumerate(prev_model.prev_tasks):
            (task_id, direction_id) = task  # The task ,direction id.
            flag = construct_flag(parser, task_id, direction_id)  # Construct the flag.
            self.prev_tasks = {i: ((task_id, direction_id), flag)}  # Construct the dictionary.

    def _distillation_loss(self, cur_out: outs_to_struct, prev_out: outs_to_struct, x: inputs_to_struct) -> torch.float:
        """
        Compute distillation loss between output of the current model and
        output of the previous (saved) model.
        Args:
            cur_out: The current output.
            prev_out: The previous output.
            x: The input.

        Returns: The distillation loss.

        """

        loss_weight = x.label_existence.unsqueeze(dim=1)  # Expand to match the shape.
        cur_out_log_softmax = torch.log_softmax(cur_out.classifier / self.temperature,
                                                dim=1)  # Compute the log-probabilities.
        prev_out_softmax = torch.softmax(prev_out.classifier / self.temperature, dim=1)  # Compute the probabilities.
        dist_loss = KLoss(cur_out_log_softmax, prev_out_softmax)  # Compute the loss.
        if self.model_flag is Flag.NOFLAG:
            dist_loss = (dist_loss * loss_weight).sum() / loss_weight.size(0)  # Count only existing characters.
        else:
            dist_loss = dist_loss.sum()
        return dist_loss

    def compute_reg_penalty(self, model: nn.Module, x: inputs_to_struct) -> torch.float:
        """
        Compute weighted distillation loss.
        Args:
            model: The model.
            x: The input.
            alpha: The regularization factor.

        Returns: The penalty.

        """
        dist_loss = 0
        old_flag = x.flag  # Store the old flag.
        for _, New_flag in self.prev_tasks.values():
            x.flag = New_flag  # Set the new flag to activate the appropriate task-head.
            y_prev = self.prev_model.forward_and_out_to_struct(x)  # The previous distribution.
            y_curr = model.forward_and_out_to_struct(x)  # The current distribution.
            dist_loss += self._distillation_loss(y_curr, y_prev, x)  # The KL div loss.
        x.flag = old_flag  # return to the original flag.
        return self.reg_factor * dist_loss

class EWC(Regulizers):
    """
    EWC plugin.
    Stores for each parameter its importance.
    """

    def __init__(self, parser: argparse,reg_type, prev_model=None, prev_data=None):
        """
        Args:
            parser: The model parser.
            mode: The training mode.
            keep_importance_data: Whether to keep the importance Data_Creation.
            prev_model: A pretrained model
            prev_data: The old dataset.
        """
        super().__init__(parser=parser, prev_model=prev_model, reg_type=reg_type)
        self.old_dataset = prev_data  # The old data-set for computing the coefficients.
        # Supporting pretrained model.
        # Update importance and old params to begin with EWC training.
      #  if os.path.exists(importance_path):
        print('Computing Importances')
        importances = self.compute_importances(prev_model, parser.bu2_loss, prev_data,
                                               parser.device, parser.train_mb_size)
        self.importances = importances
        print('Done computing Importances')
        self.saved_params = dict(copy_params_dict(prev_model.feature_extractor))  # Copy the old parameters.

    def compute_importances(self, model: nn.Module, criterion: Callable, dataset: Dataset,
                            device: str, batch_size: int) -> dict:
        """
        Compute EWC importance matrix for each parameter
        Args:
            model: The model we compute its coefficients.
            criterion: The loss criterion.
            optimizer: The optimizer.
            dataset: The dataset.
            device: The device.
            batch_size: The batch size.

        Returns: The importance coefficients.

        """
        model.eval()  # Move to evaluation mode.
        importances = zerolike_params_dict(model.feature_extractor)  # Make empty coefficients.
        dataloader = DataLoader(dataset, batch_size=batch_size)  # The dataloader.
        for i, batch in enumerate(dataloader):  # Iterating over the dataloader.
            x = preprocess(batch, device)  # Omit the ids and move to the device.
            x = self.inputs_to_struct(x)  # Make a struct.
            model.zero_grad()  # Reset grads.
            out = avalanche_forward(model, x, task_labels=None)  # Compute output.
            out = self.outs_to_struct(out)  # Make a struct.
            loss = criterion(x, out)  # Compute the loss.
            loss.backward()  # Compute grads.
            for (k1, p), (k2, imp) in zip(model.feature_extractor.named_parameters(),
                                          importances):  # Iterating over the feature weights.
                assert k1 == k2
                if p.grad is not None:
                    # Adding the grad**2.
                    imp += p.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp in importances:
            imp /= float(len(dataloader))
        # Make dictionary.
        importances = dict(importances)
        return importances

    def after_training_exp(self, model:nn.Module, dataset, optimizer) -> None:
        """
        Compute importance of parameters after each experience.
        Args:
            strategy: The strategy.

        """
        importances = self.compute_importances(
            model,
            self.parser._criterion,
            optimizer,
            dataset,
            self.device,
            self.parser.train_mb_size,
        )
        # Update importance.
        self.importances = importances
        # Update the new 'old' weights.
        self.saved_params = copy_params_dict(model.feature_extractor)

    def compute_reg_penalty(self, model:nn.Module, samples) -> None:
        """
        Compute EWC penalty and add it to the loss.
        Args:
            strategy: The strategy.

        """
        penalty = torch.tensor(0).float().to(self.device)
        cur_params = dict(model.feature_extractor.named_parameters())
        for name in self.importances.keys():
            saved_param = self.saved_params[name]  # previous weight.
            imp = self.importances[name]  # Current weight.
            cur_param = cur_params[name]
            # Add the difference to the loss.
            penalty += (imp * (cur_param - saved_param).pow(2)).sum()
        # Update the new loss.
        return self.reg_factor * penalty

class MAS(Regulizers):
    """
    Memory Aware Synapses (MAS) plugin.

    Similarly to EWC, the MAS plugin computes the importance of each
    parameter at the end of each experience. The approach computes
    importance via a second pass on the dataset. MAS does not require
    supervision and estimates importance using the gradients of the
    L2 norm of the output. Importance is then used to add a penalty
    term to the loss function.

    Technique introduced in:
    "Memory Aware Synapses: Learning what (not) to forget"
    by Aljundi et. al (2018).

    Implementation based on FACIL, as in:
    https://github.com/mmasana/FACIL/blob/master/src/approach/mas.py
    """

    def __init__(self, parser: argparse, reg_type, prev_model: Union[nn.Module, None] = None,
                 prev_data: Union[Dataset, None] = None):
        """
        Args:
            parser: The parser options.
            prev_model: The previous model.
            prev_data: The previous data.
        """

        # Init super class
        super().__init__(parser=parser, prev_model=prev_model, reg_type=reg_type)

        # Regularization Parameters and Importances parameters.
        self.batch_size, self.device, self.parser, self._lambda, self.alpha = parser.bs, parser.device, parser, \
            parser.MAS_lambda, parser.mas_alpha
        # If we have previous model we save it and compute its importances.
        self.prev_model = copy.deepcopy(prev_model)  # Previous model.
        print("Computing Importances")
        # Compute the importances.
        self.importance = self._get_importance(self.prev_model, prev_data, self.batch_size, parser.device)
        print("Done computing Importances")
        # The parameters we want to regularize are only the backbone parameters.
        self.params = self.params = dict(copy_params_dict(self.prev_model.feature_extractor))
        # Update the number of trained experiences.

    def _get_importance(self, model: nn.Module, dataset: Dataset, train_mb_size: int, device: Union['cuda', 'cpu']):
        # Initialize importance matrix for the features only.
        importance = dict(zerolike_params_dict(model.feature_extractor))
        # Do forward and backward pass to accumulate L2-loss gradients
        model.train()
        dataloader = DataLoader(dataset, batch_size=train_mb_size)
        # Progress bar
        for _, batch in enumerate(dataloader):
            # Get batch
            # Move batch to device
            batch = preprocess(batch, device)
            # Move to struct.
            batch = self.inputs_to_struct(batch)
            # Forward pass
            model.zero_grad()
            # Forward pass
            out = model.forward_and_out_to_struct(batch).classifier
            # Average L2-Norm of the output
            loss = torch.norm(out, dim=1).mean()
            loss.backward()
            # Accumulate importance
            for name, param in model.feature_extractor.named_parameters():
                if param.requires_grad:
                    if param.grad is not None:
                        importance[name] += param.grad.abs()

        # Normalize importance
        importance = {name: importance[name] / len(dataloader) for name in importance.keys()}
        return importance

    def compute_reg_penalty(self, model:nn.Module, samples) -> None:
        """
        Add the MAS loss to the classification loss.
        Args:
            strategy: The strategy.
            **kwargs:

        Returns:

        """
        # Check if the task is not the first
        loss_reg = 0.0
        # Check if properties have been initialized
        if not self.importance:
            raise ValueError("Importance is not available")
        if not self.params:
            raise ValueError("Parameters are not available")

        # Apply penalty term for each parameter.
        for name, param in model.feature_extractor.named_parameters():
            if name in self.importance.keys():
                loss_reg += torch.sum(self.importance[name] * (param - self.params[name]).pow(2))

        # Update loss
        return self.reg_factor * loss_reg

    def after_training_exp(self, model, dataset, **kwargs) -> None:
        """
        Update the Importances after the experience is finished.
        Args:
            strategy: The strategy.
            **kwargs:

        """
        print("update")
        self.params = dict(copy_params_dict(model))
        # Check if previous importance is available
        if not self.importance:
            raise ValueError("Importance is not available")
        # Get importance
        curr_importance = self._get_importance(model, dataset, self.train_mb_size,
                                               self.device)
        # Update importance
        for name in self.importance.keys():
            self.importance[name] = (self.alpha * self.importance[name] + (1 - self.alpha) * curr_importance[name])

class PI(Regulizers):
    def __init__(self, parser, prev_model, reg_type):
        super().__init__(parser = parser, prev_model = prev_model, reg_type = reg_type)
        self.big_omega = torch.zeros_like(prev_model.get_params()).to(self.device)
        self.small_omega = torch.zeros_like(prev_model.get_params()).to(self.device)


    def observe(self, model, dataset):
        dataloader = DataLoader(dataset, batch_size=self.train_mb_size)  # The dataloader.
        for i, batch in enumerate(dataloader):  # Iterating over the dataloader.
            x = preprocess(batch, self.device)  # Omit the ids and move to the device.
            x = self.inputs_to_struct(x)  # Make a struct.
            model.zero_grad()  # Reset grads.
            out = avalanche_forward(model, x, task_labels=None)  # Compute output.
            out = self.outs_to_struct(out)  # Make a struct.
            loss = self.criterion(x, out)  # Compute the loss.
            loss.backward()  # Compute grads.
            for (k1, p), (k2, imp) in zip(model.feature_extractor.named_parameters(),
                                          self.small_omega):  # Iterating over the feature weights.
                assert k1 == k2
                if p.grad is not None:
                    # Adding the grad**2.
                    imp += p.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp in self.small_omega:
            imp /= float(len(dataloader))

def Get_regulizer_according_to_reg_type(reg_type, parser, prev_model,prev_data):
    if reg_type is RegType.LFL:
        return LFL(prev_model=prev_model, parser=parser,reg_type = reg_type)
    if reg_type is RegType.LWF:
        return LWF(prev_model=prev_model, parser = parser,reg_type = reg_type)
    if reg_type is RegType.EWC:
        return EWC(prev_model=prev_model, parser = parser, reg_type = reg_type,prev_data=prev_data)
    if reg_type is RegType.MAS:
        return MAS(parser=parser, prev_model=prev_model, prev_data=prev_data, reg_type = reg_type)





