from avalanche.training.plugins import (
    SupervisedPlugin,
    EvaluationPlugin
)
#TODO - CHANGE NAMES.
from Baselines_code.avalanche_AI.training.Plugins.EWC import MyEWCPlugin
from Baselines_code.avalanche_AI.training.Plugins.LWF import MyLwFPlugin
from Baselines_code.avalanche_AI.training.Plugins.MAS import MyMASPlugin
from Baselines_code.avalanche_AI.training.Plugins.LFL import MyLFLPlugin
from Baselines_code.avalanche_AI.training.Plugins.SI import SynapticIntelligencePlugin as SIPlugin
from training.Data.Data_params import Flag
from Baselines_code.avalanche_AI.training.Plugins.RWALK import RWalkPlugin
import argparse
import sys
import os
from typing import Optional, Sequence, List, Union
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates.supervised import SupervisedTemplate
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer
from avalanche.training.plugins import LRSchedulerPlugin
from pathlib import Path
from training.Metrics.Accuracy import multi_label_accuracy_weighted, multi_label_accuracy, accuracy


class MySupervisedTemplate(SupervisedTemplate):
    def __init__(self, parser: argparse, checkpoint=None, logger=None,
                 plugins: Optional[Sequence["SupervisedPlugin"]] = None, evaluator=default_evaluator,
                 eval_every: int = -1):
        """
        Args:
            parser: The parser.
            checkpoint: The checkpoint.
            device: The device.
            logger: The logger.
            plugins: Possible plugins.
            evaluator: The evaluator.
            eval_every: Interval evaluation.
            **base_kwargs: Optional args
        """
        self.task_id = 0
        self.direction_id = 0
        self.parser = parser  # The parser.
        self.logger = logger  # The logger.
        self.checkpoint = checkpoint  # The Checkpoint
        self.inputs_to_struct = parser.inputs_to_struct
        self.outs_to_struct = parser.outs_to_struct
        self.scheduler = LRSchedulerPlugin(parser.scheduler, reset_scheduler=False, reset_lr=False,
                                           step_granularity='iteration')  # Every iteration updatable scheduler.
        plugins.append(self.scheduler)  # Add the scheduler to the Plugins.

        super().__init__(
            model=parser.model,
            optimizer=parser.optimizer,
            criterion=parser.criterion,
            train_mb_size=parser.train_mb_size,
            train_epochs=parser.train_epochs,
            eval_mb_size=parser.eval_mb_size,
            device=parser.device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every
        )

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
        Make output struct.
        """
        out = self.parser.outs_to_struct(self.mb_output)
        return self._criterion(self.parser, self.mb_x, out)

    def eval_epoch(self, **kwargs):
        """Evaluation loop over the current `self.dataloader`."""
        super().eval_epoch(**kwargs)
        Metrics = self.evaluator.get_last_metrics()
      #  print(Metrics.keys())
        if Metrics == {}:
            acc = 0.0
        else:
            for key in Metrics.keys():
                if 'Top1_Acc_Exp/eval_phase/test_stream' in key:
                 acc = Metrics[key]
                 print("The Accuracy is: {}".format(acc))
        # TODO -Get the task, direction from somewhere.
        self.checkpoint(self.model, self.clock.train_exp_epochs, acc, self.optimizer, self.parser.scheduler, self.parser,0 , 2)  # Updating checkpoint.
     #   print(self.evaluator.get_all_metrics())

    def update_task(self, task_id: int, direction_id: int):
        self.task_id = task_id
        self.direction_id = direction_id


class MyEWC(MySupervisedTemplate):
    def __init__(
            self,
            parser: argparse,
            mode: str = "separate",
            decay_factor: Optional[float] = None,
            keep_importance_data: bool = False,
            plugins: Optional[List[SupervisedPlugin]] = None,
            evaluator=default_evaluator,
            eval_every=-1,
            prev_model= None,
            **base_kwargs
    ):
        """
        Args:
            parser: The parser.
            mode: The EWC mode 'online' or 'separate'.
            decay_factor: The decay factor.
            keep_importance_data:
            device: The device.
            plugins: The optional plugins.
            evaluator: The evaluator.
            eval_every: The evaluation interval.
            **base_kwargs: Optional args.
        """
        ewc = MyEWCPlugin(parser, mode, decay_factor, keep_importance_data, prev_model, parser.old_dataset)
        if plugins is None:
            plugins = [ewc]
        else:
            plugins.append(ewc)

        super().__init__(
            parser,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )


class LWF(MySupervisedTemplate):
    """
    Learning without forgetting.
    """

    def __init__(
            self,
            parser: argparse,
            plugins: Optional[List[SupervisedPlugin]] = None,
            evaluator: EvaluationPlugin = default_evaluator,
            eval_every: int = -1,
            prev_model = None,
            **base_kwargs
    ):
        """
        Args:
           parser: The parser.
           plugins: The optional plugins.
           evaluator: The evaluator.
           eval_every: The interval evaluation.
           **base_kwargs: Optional args.
        """
        lwf = MyLwFPlugin(parser, prev_model=prev_model)
        if plugins is None:
            plugins = [lwf]
        else:
            plugins.append(lwf)

        super().__init__(
            parser,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )


class LFL(MySupervisedTemplate):
    """Less Forgetful Learning strategy.

    See LFL plugin for details.
    Refer Paper: https://arxiv.org/pdf/1607.00122.pdf
    This strategy does not use task identities.
    """

    def __init__(
            self,
            parser: argparse,
            plugins: Optional[List[SupervisedPlugin]] = None,
            evaluator: EvaluationPlugin = default_evaluator,
            eval_every: int = -1,
            prev_model = None,
            **base_kwargs
    ):
        """
        Args:
           parser: The parser.
           plugins: Optional plugins.
           evaluator: The evaluator.
           eval_every: Evaluation interval.
           **base_kwargs:
        """
        lfl = MyLFLPlugin(parser.LFL_lambda, prev_model)
        if plugins is None:
            plugins = [lfl]
        else:
            plugins.append(lfl)

        super().__init__(
            parser,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )


class MyMAS(MySupervisedTemplate):
    """Less Forgetful Learning strategy.

    See LFL plugin for details.
    Refer Paper: https://arxiv.org/pdf/1607.00122.pdf
    This strategy does not use task identities.
    """

    def __init__(self, parser: argparse, device: str = 'cuda', plugins: Optional[List[SupervisedPlugin]] = None,
                 evaluator: EvaluationPlugin = default_evaluator, eval_every: int = -1, prev_mode = None, **base_kwargs):
        """
       Args:
          parser: The parser.
          plugins: Optional plugins.
          evaluator: The evaluator.
          eval_every: Evaluation interval.
          prev_model
          **base_kwargs:
       """

        MAS = MyMASPlugin(parser, prev_mode)
        if plugins is None:
            plugins = [MAS]
        else:
            plugins.append(MAS)

        super().__init__(
            parser,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )


class MyRWALK(MySupervisedTemplate):
    def __init__(
            self,
            parser: argparse,
            device: str = 'cuda',
            plugins: Optional[List[SupervisedPlugin]] = None,
            evaluator: EvaluationPlugin = default_evaluator,
            eval_every: int = -1,
            **base_kwargs
    ):
        """
        Args:
          parser: The parser.
          plugins: Optional plugins.
          evaluator: The evaluator.
          eval_every: Evaluation interval.
          **base_kwargs:
       """

        RWalk = RWalkPlugin(parser)
        if plugins is None:
            plugins = [RWalk]
        else:
            plugins.append(RWalk)

        super().__init__(
            parser,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )

class SI(MySupervisedTemplate):
    def __init__(self, parser: argparse, device: str = 'cuda', plugins: Optional[List[SupervisedPlugin]] = None,
                 evaluator: EvaluationPlugin = default_evaluator, eval_every: int = -1, exc = None, **base_kwargs):
        """
       Args:
          parser: The parser.
          plugins: Optional plugins.
          evaluator: The evaluator.
          eval_every: Evaluation interval.
          prev_model
          **base_kwargs:
       """

        SIP = SIPlugin(parser.si_lambda,excluded_parameters = None)
        if plugins is None:
            plugins = [SIP]
        else:
            plugins.append(SIP)

        super().__init__(
            parser,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )