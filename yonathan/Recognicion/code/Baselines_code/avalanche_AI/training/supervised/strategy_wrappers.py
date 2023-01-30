"""
Here we define the strategy,
to support all regularization methods.
"""
import argparse
import os.path
from pathlib import Path

import torch
import torch.optim as optim
from Baselines_code.avalanche_AI.training.Plugins.Evaluation import accuracy_metrics
from Baselines_code.avalanche_AI.training.Plugins.classes import Get_regularization_plugin
from Baselines_code.avalanche_AI.training.Plugins.plugins_base import Base_plugin
from Baselines_code.baselines_utils import RegType
from avalanche.benchmarks.generators import dataset_benchmark
from avalanche.evaluation.metrics import loss_metrics
from avalanche.logging import WandBLogger, InteractiveLogger
from avalanche.training.plugins import LRSchedulerPlugin
from avalanche.training.plugins.evaluation import EvaluationPlugin
from avalanche.training.templates.supervised import SupervisedTemplate
from training.Data.Checkpoints import CheckpointSaver
from training.Data.Structs import outs_to_struct
from training.Utils import create_optimizer_and_scheduler


class Regularization_strategy(SupervisedTemplate):
    """
    The basic Strategy, every strategy inherits from.
    """

    def __init__(self, opts: argparse, reg_type: RegType, task: tuple[int, tuple[int, int]], logger=None,
                 eval_every: int = -1, prev_model=None, model_path=None):
        """
        Args:
            opts: The model opts.
            task: The task.
            logger: The logger.
            eval_every: Interval evaluation.
            prev_model: The previous model.
            model_path: The model path.
        """
        plugins = []
        self.Project_path = Path(__file__).parents[5]  # The project path.
        self.task = task  # The task.
        self.reg_type = reg_type  # The regularization type.
        self.opts = opts  # The model opts.
        self.logger = logger  # The logger.
        self.ds_type = opts.ds_type
        self.inputs_to_struct = opts.inputs_to_struct
        self.outs_to_struct = opts.outs_to_struct
        self.load_from = model_path
        self.scheduler = None
        self.Model_folder = self.Get_Model_folder()  # The model folder.
        self.checkpoint = CheckpointSaver(self.Model_folder)  # The Checkpoint.
        loggers, evaluator = self.Get_logger_and_evaluator()
        if self.reg_type is not RegType.Naive:
            self.regularization: Base_plugin = Get_regularization_plugin(prev_checkpoint=prev_model,
                                                                         load_from=self.load_from, opts=self.opts,
                                                                         reg_type=self.reg_type)
            plugins.append(self.regularization)

        super(Regularization_strategy, self).__init__(
            model=opts.model,
            optimizer=optim.Adam(opts.model.parameters()),
            criterion=opts.criterion,
            train_mb_size=opts.train_mb_size,
            train_epochs=opts.train_epochs,
            eval_mb_size=opts.eval_mb_size,
            device=opts.device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every
        )

    def make_optimizer(self, **kwargs):
        """
        We already passed the optimizer in the initialization.
        Now we update to take only specific parameters.
        """
        (new_task, dl_len) = kwargs['kargs']
        self.update_task(new_task=new_task)
        learned_params = self.model.get_specific_head(new_task[0], new_task[1])  # Train only the desired params.
        #   learned_params = self.model.parameters()
        self.optimizer, self.scheduler = create_optimizer_and_scheduler(self.opts, learned_params,
                                                                        nbatches=dl_len // self.opts.bs)
        scheduler = LRSchedulerPlugin(self.scheduler, reset_scheduler=False, reset_lr=False,
                                      step_granularity='iteration')  # Every iteration updatable scheduler.
        self.plugins.append(scheduler)  # Add the scheduler to the Plugins.

    def update_task(self, new_task: tuple):
        """
        Args:
            new_task: The new task

        Returns:

        """
        self.model.update_task_list(new_task)

    @property
    def mb_x(self):
        """
        Current mini-batch input.
        """
        return self.inputs_to_struct(self.mbatch)

    def forward(self) -> outs_to_struct:
        """
        Forward the model.
        Returns: The model output.

        """
        return self.model.forward_and_out_to_struct(self.mb_x)

    def criterion(self) -> torch.float:
        """
        Loss function.
        Make output to struct.
        """
        return self._criterion(self.opts, self.mb_x, self.mb_output)

    def train_sequence(self, scenario: dataset_benchmark):
        """

        Args:
            scenario:
        """
        for (exp_train, exp_test) in zip(scenario.train_stream, scenario.test_stream):
            self.train(exp_train, [exp_test], kargs=(self.task, len(exp_train.dataset)))
            print("Done training, now final evaluation of the model.")
            self.eval(exp_test)

    def eval_epoch(self, **kwargs: dict) -> None:
        """
        Evaluation loop over the current `self.dataloader`.
        Args:
            **kwargs: The args.

        Returns: Saves the new model and updated the optimum if needed.

        """
        super(Regularization_strategy, self).eval_epoch(**kwargs)
        if self.clock.train_iterations > 0:
            Metrics = self.evaluator.get_last_metrics()
            Metrics = dict(
                filter(lambda key: key[0].startswith('Top1_Acc_Exp/eval_phase/test_stream'), Metrics.items()))
            acc = list(Metrics.items())[-1][-1]
            reg_state_dict = self.state_dict()
            new_key = "reg_state_dict"
            new_value = (new_key, reg_state_dict)
            self.checkpoint(self.model, self.clock.train_exp_epochs, acc, self.optimizer, self.scheduler,
                            self.opts, new_value)  # Updating checkpoint.

    def state_dict(self):
        """
        Returns:
        """
        strategy_state_dict = dict()
        strategy_state_dict['current_data_loader'] = self.dataloader
        strategy_state_dict['regulizer_state_dict'] = self.regularization.state_dict(
            self) if self.reg_type is not RegType.Naive else None
        strategy_state_dict['current_model'] = self.model.state_dict()
        strategy_state_dict['trained_tasks'] = self.model.trained_tasks
        return strategy_state_dict

    def Get_Model_folder(self):
        """
        Returns:
        """
        # Path to the data-set.
        Data_specific_path = os.path.join(self.Project_path, 'data/{}'.format(str(self.ds_type)))
        # Path to the results.
        results_path = os.path.join(Data_specific_path, f'Baselines/')
        # Path to the regularization type results.
        Model_folder = os.path.join(results_path,
                                    f"Model_{self.task}/{str(self.reg_type)}/lambda"
                                    f"={str(self.reg_type.class_to_reg_factor(self.opts))}")
        return Model_folder

    def Get_logger_and_evaluator(self):
        """
        Returns:

        """
        log_path = os.path.join(self.Project_path, f'data/Baselines/{str(self.reg_type)}/logging')
        Checkpoint_path = os.path.join(self.Project_path,
                                       f'fata/Baselines/{str(self.reg_type)}/checkpoints/Task_{self.task[0]}_'
                                       f'{self.task[1]}')
        loggers = [WandBLogger(project_name=f"avalanche_{str(self.reg_type)}", run_name="train", dir=log_path,
                               path=Checkpoint_path), InteractiveLogger()]
        evaluator = EvaluationPlugin(accuracy_metrics(self.opts, minibatch=True, epoch=True, experience=True,
                                                      stream=True),
                                     loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
                                     loggers=loggers)
        return loggers, evaluator


__all__ = [
    "Regularization_strategy"
]
