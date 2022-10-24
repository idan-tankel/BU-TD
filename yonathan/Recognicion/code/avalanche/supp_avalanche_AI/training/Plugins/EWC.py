import sys
import warnings

sys.path.append(r'/')
from supp.general_functions import preprocess
from torch.utils.data import DataLoader
import torch
from avalanche.models.utils import avalanche_forward
from avalanche.training.utils import copy_params_dict, zerolike_params_dict
from avalanche.training.plugins import EWCPlugin

class MyEWCPlugin(EWCPlugin):
    def __init__( self,parser, mode="separate",   decay_factor=None,   keep_importance_data=False,prev_model = None, old_dataset = None ):
        """
        Args:
            parser: The model parser.
            mode: The training mode.
            decay_factor:
            keep_importance_data:
            prev_model: A pretrained model
            old_dataset:
        """
        super().__init__( ewc_lambda = parser.ewc_lambda, mode = mode, decay_factor = decay_factor,  keep_importance_data = keep_importance_data)
        self.old_dataset = old_dataset
        self.prev_model = prev_model
        self.parser = parser
        # Supporting pretrained model.
        if prev_model != None and old_dataset != None:
            # Update importances and old params to begin with EWC training.
            print("Computing Importances")
            importances = self.compute_importances(prev_model, parser.criterion, parser.optimizer, old_dataset, parser.device, parser.train_mb_size, False)
            self.update_importances(importances, 0)
            print("Done computing Importances")
            self.saved_params[0] = dict(copy_params_dict(prev_model.bumodel)) # Change to the excluded params.

    def compute_importances(self, model, criterion, optimizer, dataset, device, batch_size,use_task_ids):
        """
        Compute EWC importance matrix for each parameter
        """
        model.eval()
       # Ignored_params = self.Ignored_params
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
        # TODO - CHANGE TO FE.
        importances = zerolike_params_dict(model.bumodel)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        for i, batch in enumerate(dataloader):
            if use_task_ids:
             x = preprocess(batch[:-1],device)
            else:
             x = preprocess(batch, device)
            task_labels = batch[-1].to(device)
            if len(x[1].shape) == 1:
             x[1] = x[1].view([-1,1])
            optimizer.zero_grad()
            out = avalanche_forward(model, x, task_labels)
            loss = criterion(self.parser, x, out)
            loss.backward()

            for (k1, p), (k2, imp) in zip(model.bumodel.named_parameters(), importances):
                assert k1 == k2
                if p.grad is not None:
                    imp += p.grad.data.clone().pow(2)

        # average over mini batch length

        for _, imp in importances:
            imp /= float(len(dataloader))

        importances = dict(importances)

        return importances

    def after_training_exp(self, strategy, **kwargs):
        """
        Compute importance of parameters after each experience.
        """

        exp_counter = strategy.EpochClock.train_exp_counter
        if strategy.EpochClock.train_exp_epochs == strategy.EpochClock.max_epochs:
            importances = self.compute_importances(
                strategy.model,
                strategy._criterion,
                strategy.optimizer,
                strategy.experience.dataset,
                strategy.device,
                strategy.train_mb_size,
                True
            )
            self.update_importances(importances, exp_counter)
            self.saved_params[exp_counter] = copy_params_dict(strategy.model.bumodel)
            # clear previous parameter values
            if exp_counter > 0 and (not self.keep_importance_data):
                del self.saved_params[exp_counter - 1]

    def before_backward(self, strategy, **kwargs):
        """
        Compute EWC penalty and add it to the loss.
        # TODO - SUPPORT ALL SUPPORTED THINGS LIKE EXP_COUNTER = 0, MULTITASKLOSS.
        """
        exp_counter = strategy.EpochClock.train_exp_counter
        if  exp_counter == 0 or self.ewc_lambda == 0.0:
            return
        penalty = torch.tensor(0).float().to(strategy.device)
        if self.mode == "separate":
            for experience in range(1): # TODO - CHANGE TO ALL TASKS.
                Cur_params = dict(strategy.model.bumodel.named_parameters())
                for name in self.importances[0].keys():
                    # TODO - CHANGE IT to general index in the future according to some policy!!!!
                    saved_param = self.saved_params[0][name]
                    imp = self.importances[0][name]
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
      #  print(self.ewc_lambda * penalty)
        strategy.loss += self.ewc_lambda * penalty