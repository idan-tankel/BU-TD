import warnings
import torch
from supp.data_functions import preprocess
from avalanche.training.utils import copy_params_dict, zerolike_params_dict
from torch.utils.data import DataLoader
from avalanche.models import avalanche_forward
import copy
import torch
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.utils import get_last_fc_layer, freeze_everything
from avalanche.models.base_model import BaseModel
from avalanche.training.plugins import (
    SupervisedPlugin,
    CWRStarPlugin,
    ReplayPlugin,
    GenerativeReplayPlugin,
    TrainGeneratorAfterExpPlugin,
    GDumbPlugin,
    LwFPlugin,
    AGEMPlugin,
    GEMPlugin,
    EWCPlugin,
    EvaluationPlugin,
    SynapticIntelligencePlugin,
    CoPEPlugin,
    GSS_greedyPlugin,
    LFLPlugin,
    MASPlugin,
)

from avalanche.training.plugins.clock import Clock

class MyEWCPlugin(EWCPlugin):
    def __init__(
            self,
            ewc_lambda,
            start_from_regulization=False,
            Ignored_params = None,
            mode="separate",
            decay_factor=None,
            keep_importance_data=False,
    ):

     super().__init__( ewc_lambda = ewc_lambda, mode = mode, decay_factor = decay_factor,  keep_importance_data = keep_importance_data)
     self.start_from_regulization = start_from_regulization
     self.Ignored_params = Ignored_params


    def compute_importances(self, model, criterion, optimizer, dataset, device, batch_size):
        """
        Compute EWC importance matrix for each parameter
        """
        model.eval()
        Ignored_params = self.Ignored_params
        # Set RNN-like modules on GPU to training mode to avoid CUDA error
        if device == "cuda":
            for module in model.modules():
                if isinstance(module, torch.nn.RNNBase):
                    warnings.warn(
                        "RNN-like modules do not support "
                        "backward calls while in `eval` mode on CUDA "
                        "devices. Setting all `RNNBase` modules to "
                        "`train` mode. May produce inconsistent "
                        "output if such modules have `dropout` > 0."
                    )
                    module.train()

        # list of list
        importances = zerolike_params_dict(model)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        for i, batch in enumerate(dataloader):
            x = preprocess(batch[:-1])
            task_labels = batch[-1].to(device)
            if len(x[1].shape)  == 1:
             x[1] = x[1].view([-1,1])
            optimizer.zero_grad()
            out = avalanche_forward(model, x, task_labels)
            loss = criterion(x, out)
            loss.backward()

            for (k1, p), (k2, imp) in zip(
                    model.named_parameters(), importances
            ):
                assert k1 == k2
                if p.grad is not None:
                    imp += p.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp in importances:
            imp /= float(len(dataloader))
        s = 0
        importances_without_taskhead = {}
        for i in range(len(importances)):
          (name, param ) = importances[i]
          if not 'taskhead.taskhead' in name:
              importances_without_taskhead[name] = param


        return importances_without_taskhead

    def before_training_exp(self,strategy, **kwargs):
        if self.start_from_regulization:
            exp_counter = 0
            importances = self.compute_importances(
                strategy.model,
                strategy._criterion,
                strategy.optimizer,
                strategy.experience.dataset,
                strategy.device,
                strategy.train_mb_size,
            )
            self.update_importances(importances, exp_counter)
            self.saved_params[exp_counter] = dict(copy_params_dict(strategy.model))
          #  self.start_from_regulization = False

    def after_training_exp(self, strategy, **kwargs):
        """
        Compute importances of parameters after each experience.
        """
        if self.start_from_regulization:
         exp_counter = strategy.clock.train_exp_counter + 1
        else:
         exp_counter = strategy.clock.train_exp_counter

        importances = self.compute_importances(
            strategy.model,
            strategy._criterion,
            strategy.optimizer,
            strategy.experience.dataset,
            strategy.device,
            strategy.train_mb_size,
        )
        self.update_importances(importances, exp_counter)
        self.saved_params[exp_counter] = copy_params_dict(strategy.model)
        # clear previous parameter values
        if exp_counter > 0 and (not self.keep_importance_data):
            del self.saved_params[exp_counter - 1]

    def before_backward(self, strategy, **kwargs):
        """
        Compute EWC penalty and add it to the loss.
        """
        exp_counter = strategy.clock.train_exp_counter
        if exp_counter == 0 and not self.start_from_regulization:
            return
     #   print("success")
        exp_counter +=1
        penalty = torch.tensor(0).float().to(strategy.device)
        if self.mode == "separate":
            for experience in range(exp_counter):
                '''
                for (_, cur_param), (_, saved_param), (_, imp) in zip(
                    strategy.model.named_parameters(),
                    self.saved_params[experience],
                    self.importances[experience],
                ):
                '''
                Cur_params = dict(strategy.model.named_parameters())
                for name in self.importances[experience].keys():
                    saved_param = self.saved_params[exp_counter-1][name]
                    imp = self.importances[experience][name]
                    cur_param = Cur_params[name]
                    # dynamic models may add new units
                    # new units are ignored by the regularization
                    n_units = saved_param.shape[0]
                    cur_param = cur_param[:n_units]
                    penalty += (imp * (cur_param - saved_param).pow(2)).sum()
        elif self.mode == "online":
            prev_exp = exp_counter - 1
            for (_, cur_param), (_, saved_param), (_, imp) in zip(
                strategy.model.named_parameters(),
                self.saved_params[prev_exp],
                self.importances[prev_exp],
            ):
                # dynamic models may add new units
                # new units are ignored by the regularization
                n_units = saved_param.shape[0]
                cur_param = cur_param[:n_units]
                penalty += (imp * (cur_param - saved_param).pow(2)).sum()
        else:
            raise ValueError("Wrong EWC mode.")

        strategy.loss += self.ewc_lambda * penalty

class MylwfPlugin(LwFPlugin):
    def _distillation_loss(self, out, prev_out):
        """
        Compute distillation loss between output of the current model and
        and output of the previous (saved) model.
        """
        # we compute the loss only on the previously active units.
      #  au = list(active_units)

        task_out = out[1]
        prev_out_task = prev_out[1]
        log_p = torch.log_softmax(task_out / self.temperature, dim=1)
        q = torch.softmax(prev_out_task / self.temperature, dim=1)
        res = torch.nn.functional.kl_div(log_p, q, reduction="batchmean")
        return res

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


    def compute_features(self, model, x):
        """
        Compute features from prev model and current model
        """
        model.eval()
        self.prev_model.eval()

        features = model(x)[-1]
        prev_features = self.prev_model(x)[-1]

        return features, prev_features

    def after_training_exp(self, strategy, **kwargs):
        """
        Save a copy of the model after each experience
        and freeze the prev model and freeze the last layer of current model
        """

        self.prev_model = copy.deepcopy(strategy.model)

        freeze_everything(self.prev_model)

      #  last_fc_name, last_fc = get_last_fc_layer(strategy.model)

     #   for param in last_fc.parameters():
     #       param.requires_grad = False

    def before_training(self, strategy, **kwargs):
        """
        Check if the model is an instance of base class to ensure get_features()
        is implemented 
        """ 
        return None          


class MyClock(Clock):
    def __init__(self,iter = 0):
     super().__init__()
     self.train_exp_counter = iter


