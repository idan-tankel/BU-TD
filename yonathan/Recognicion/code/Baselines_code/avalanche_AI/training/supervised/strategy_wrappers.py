import argparse
import os.path
from pathlib import Path

import torch.nn as nn
import torch.optim as optim
from avalanche.evaluation.metrics import loss_metrics
from avalanche.logging import WandBLogger, InteractiveLogger
from avalanche.training.plugins import LRSchedulerPlugin
from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.plugins.evaluation import EvaluationPlugin
from avalanche.training.templates.supervised import SupervisedTemplate
from torch.utils.data import DataLoader

from Baselines_code.avalanche_AI.training.Plugins.EWC import EWC
from Baselines_code.avalanche_AI.training.Plugins.Evaluation import accuracy_metrics
from Baselines_code.avalanche_AI.training.Plugins.IMM_Mean import IMM_Mean as IMM_Mean
from Baselines_code.avalanche_AI.training.Plugins.IMM_Mode import MyIMM_Mode as IMM_Mode
from Baselines_code.avalanche_AI.training.Plugins.LFL import LFL
from Baselines_code.avalanche_AI.training.Plugins.LWF import LwF
from Baselines_code.avalanche_AI.training.Plugins.MAS import MAS
from Baselines_code.avalanche_AI.training.Plugins.SI import SI
from training.Data.Checkpoints import CheckpointSaver
from training.Data.Data_params import RegType
from training.Utils import create_optimizer_and_scheduler


class Regularization_strategy(SupervisedTemplate):
    """
    The basic Strategy, every strategy inherits from.
    """

    def __init__(self, parser: argparse, task=[0, (1, 0)], logger=None,
                 eval_every: int = -1, prev_model = None, prev_data=None, model_path=None):
        """
        Args:
            parser: The parser.
            checkpoint: The checkpoint.
            task: The task.
            logger: The logger.
            eval_every: Interval evaluation.
        """
        self.task_id = task[0]
        self.direction_id = task[1]
        self.parser = parser  # The parser.
        self.logger = logger  # The logger.
        self.inputs_to_struct = parser.inputs_to_struct
        self.outs_to_struct = parser.outs_to_struct
        self.reg_type = parser.reg_type
        self.load_from = model_path
        self.scheduler = None
        plugins = []
        Project_path = Path(__file__).parents[5]
        #
      #  project_path = Path(__file__).parents[2]
        # Path to the data-set.
        Data_specific_path = os.path.join(Project_path, 'data/{}'.format(str(parser.ds_type)))
        # Path to the results.
        results_path = os.path.join(Data_specific_path, f'Baselines/')
        # Path to the regularization type results.
        Model_folder = os.path.join(results_path,
                                    f"{str(parser.reg_type)}SGD/Task_{task}/lambda"
                                    f"={str(parser.reg_type.class_to_reg_factor(parser))}")
        self.Model_folder = Model_folder
        self.checkpoint = CheckpointSaver(Model_folder) # The Checkpoint.
        #
        log_path = os.path.join(Project_path, f'data/Baselines/{str(self.reg_type)}/logging')
        Checkpoint_path = os.path.join(Project_path,
                                       f'fata/Baselines/{str(self.reg_type)}/checkpoints/Task_{task[0]}_{task[1]}')
        loggers = [WandBLogger(project_name=f"avalanche_{str(self.reg_type)}", run_name="train", dir=log_path,
                               path=Checkpoint_path), InteractiveLogger()]
        evaluator = EvaluationPlugin(accuracy_metrics(parser, minibatch=True, epoch=True, experience=True, stream=True),
                                     loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
                                     loggers=loggers)

        if self.reg_type is not RegType.Naive:
            plugins.append(self.Get_regularization_plugin(prev_model, prev_data))

        super().__init__(
            model=parser.model,
            optimizer=optim.Adam(parser.model.parameters()),
            criterion=parser.criterion,
            train_mb_size=parser.train_mb_size,
            train_epochs=parser.train_epochs,
            eval_mb_size=parser.eval_mb_size,
            device=parser.device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every
        )

    def make_optimizer(self, **kwargs):
        """
        We already pass the optimizer in the initialization.
        """
        (new_task, len) = kwargs['kargs']
        learned_params = self.model.get_specific_head(new_task[0], new_task[1])  # Train only the desired params.
        self.optimizer, self.scheduler= create_optimizer_and_scheduler(self.parser, learned_params, nbatches = len //
                                                                                                      self.parser.bs)
        scheduler = LRSchedulerPlugin(self.scheduler, reset_scheduler=False, reset_lr=False,
                                           step_granularity='iteration')  # Every iteration updatable scheduler.
        self.plugins.append(scheduler)  # Add the scheduler to the Plugins.

    @property
    def mb_x(self):
        """
        Current mini-batch input.
        Omit the ids and make a struct.
        """
        return self.parser.inputs_to_struct(self.mbatch[:-1])

    def criterion(self):
        """
        Loss function.
        Make output to struct.
        """
        out = self.parser.outs_to_struct(self.mb_output)
        return self._criterion(self.parser, self.mb_x, out)

    def eval_epoch(self, **kwargs):
        """Evaluation loop over the current `self.dataloader`."""
        super().eval_epoch(**kwargs)
        if self.clock.train_iterations > 0:
            Metrics = self.evaluator.get_last_metrics()
            Metrics = dict(filter(lambda key: key[0].startswith('Top1_Acc_Exp/eval_phase/test_stream'), Metrics.items()))
            try:
                acc = list(Metrics.items())[-1][-1]
                print(f"The Accuracy is: {acc}")
            except IndexError:
                acc = 0.0

            self.checkpoint(self.model, self.clock.train_exp_epochs, acc, self.optimizer, self.scheduler,
                            self.parser, self.task_id, self.direction_id)  # Updating checkpoint.

    def update_task(self, task_id: int, direction_id: int) -> None:
        """
        Updates the task, direction id.
        Args:
            task_id: The task id.
            direction_id: The direction id.

        """
        self.task_id = task_id
        self.direction_id = direction_id

    def Get_regularization_plugin(self, prev_model: nn.Module, prev_data: dict[DataLoader]) -> SupervisedPlugin:
        """
        Returns the desired regularization plugin.
        Args:
            prev_model: The previous model.
            prev_data: The previous data.

        Returns: Regularization plugin.

        """
        if self.reg_type is RegType.EWC:
            return EWC(parser=self.parser, prev_model=prev_model, old_dataset=prev_data['train_ds'],
                       load_from=self.load_from)
        if self.reg_type is RegType.LFL:
            return LFL(self.parser, prev_model)
        if self.reg_type is RegType.LWF:
            return LwF(self.parser, prev_model)
        if self.reg_type is RegType.MAS:
            return MAS(parser=self.parser, prev_model=prev_model, prev_data=prev_data['train_ds'],
                       load_from=self.load_from)
        if self.reg_type is RegType.IMM_Mean:
            return IMM_Mean(parser=self.parser, prev_model=prev_model)
        if self.reg_type is RegType.IMM_Mode:
            return IMM_Mode(parser=self.parser, prev_model=prev_model, old_dataset=prev_data['train_ds'],
                            load_from=self.load_from)
        if self.reg_type is RegType.SI:
            return SI(parser=self.parser, prev_model=prev_model)

        else:
            raise NotImplementedError


__all__ = [
    "Regularization_strategy"
]
