from avalanche.training.plugins import SupervisedPlugin, EvaluationPlugin
from Baselines_code.avalanche_AI.training.Plugins.EWC import MyEWCPlugin
from Baselines_code.avalanche_AI.training.Plugins.LWF import MyLwFPlugin
from Baselines_code.avalanche_AI.training.Plugins.MAS import MyMASPlugin
from Baselines_code.avalanche_AI.training.Plugins.LFL import MyLFLPlugin
from Baselines_code.avalanche_AI.training.Plugins.SI import SynapticIntelligencePlugin as SIPlugin
from Baselines_code.avalanche_AI.training.Plugins.RWALK import RWalkPlugin
import argparse
from typing import Optional, Sequence, List
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates.supervised import SupervisedTemplate
from avalanche.training.plugins import LRSchedulerPlugin


class MySupervisedTemplate(SupervisedTemplate):
    """
    The basic Strategy, every strategy inherits from.
    """

    def __init__(self, parser: argparse, checkpoint=None, task=[0, (1, 0)], logger=None,
                 plugins: Optional[Sequence["SupervisedPlugin"]] = [], evaluator=default_evaluator,
                 eval_every: int = -1):
        """
        Args:
            parser: The parser.
            checkpoint: The checkpoint.
            task: The task.
            logger: The logger.
            plugins: Possible plugins.
            evaluator: The evaluator.
            eval_every: Interval evaluation.
        """
        self.task_id = task[0]
        self.direction_id = task[1]
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
        Make output to struct.
        """
        out = self.parser.outs_to_struct(self.mb_output)
        return self._criterion(self.parser, self.mb_x, out)

    def eval_epoch(self, **kwargs):
        """Evaluation loop over the current `self.dataloader`."""
        super().eval_epoch(**kwargs)
        Metrics = self.evaluator.get_last_metrics()
        acc = 0.0
        for key in Metrics.keys():
            if 'Top1_Acc_Exp/eval_phase/test_stream' in key:
                acc = Metrics[key]
                print("The Accuracy is: {}".format(acc))

        self.checkpoint(self.model, self.clock.train_exp_epochs, acc, self.optimizer, self.parser.scheduler,
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


class Naive(MySupervisedTemplate):
    """
    Naive strategy, without any regularization.
    Used for training the initial task.
    """

    def __init__(
            self,
            parser: argparse,
            plugins: Optional[List[SupervisedPlugin]] = [],
            evaluator=default_evaluator,
            eval_every=-1,
            **base_kwargs
    ):
        super().__init__(
            parser,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )


class MyEWC(MySupervisedTemplate):
    """
    EWC strategy.
    """

    def __init__(
            self,
            parser: argparse,
            mode: str = "separate",
            keep_importance_data: bool = False,
            plugins: Optional[List[SupervisedPlugin]] = None,
            evaluator=default_evaluator,
            eval_every=-1,
            prev_model=None,
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
        ewc = MyEWCPlugin(parser=parser, mode=mode, keep_importance_data=keep_importance_data, prev_model=prev_model,
                          prev_data=parser.old_dataset)
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
            prev_model=None,
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
            prev_model=None,
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

    def __init__(self, parser: argparse, plugins: Optional[List[SupervisedPlugin]] = None,
                 evaluator: EvaluationPlugin = default_evaluator, eval_every: int = -1, prev_model=None, prev_data=None,
                 **base_kwargs):
        """
       Args:
          parser: The parser.
          plugins: Optional plugins.
          evaluator: The evaluator.
          eval_every: Evaluation interval.
          prev_model: The previous model.
          **base_kwargs:
       """

        MAS = MyMASPlugin(parser, prev_model, prev_data)
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


''''
class MyRWALK(MySupervisedTemplate):
    def __init__(
            self,
            parser: argparse,
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

        super(MySupervisedTemplate).__init__(
            parser,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )


class SI(MySupervisedTemplate):
    def __init__(self, parser: argparse, plugins: Optional[List[SupervisedPlugin]] = None,
                 evaluator: EvaluationPlugin = default_evaluator, eval_every: int = -1, **base_kwargs):
        """
       Args:
          parser: The parser.
          plugins: Optional plugins.
          evaluator: The evaluator.
          eval_every: Evaluation interval.
          prev_model
          **base_kwargs:
       """

        SIP = SIPlugin(parser.si_lambda, excluded_parameters=None)
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

'''

__all__ = [
    "LFL",
    "LWF",
    "MyMAS",
    "MyEWC",
    "Naive"
]
