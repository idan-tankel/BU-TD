import argparse
import sys
from typing import Optional, Sequence, List, Union

from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates.supervised import SupervisedTemplate
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer

sys.path.append(r'/home/sverkip/data/BU-TD/yonathan/Recognicion/code/Baselines_code')
from Baselines_code.avalanche_AI.training.Plugins.EWC import MyEWCPlugin
from Baselines_code.avalanche_AI.training.Plugins.LWF  import MyLwFPlugin
from Baselines_code.avalanche_AI.training.Plugins.MAS import MyMASPlugin
from Baselines_code.avalanche_AI.training.Plugins.LFL  import MyLFLPlugin
from Baselines_code.avalanche_AI.training.Plugins.Clock import EpochClock
from Baselines_code.avalanche_AI.training.Plugins.SI  import SynapticIntelligencePlugin as MySIPlugin
from Baselines_code.avalanche_AI.training.Plugins.RWALK import RWalkPlugin
from Baselines_code.avalanche_AI.training.Plugins.AGEM import MyAGEMPlugin
from Baselines_code.avalanche_AI.training.Plugins.IL import IL
from Baselines_code.avalanche_AI.training.Plugins.GEM import GEMPlugin as MyGEMPlugin
from supp.loss_and_accuracy import multi_label_accuracy_weighted # TODO - GIT RID OF THOSE STUPID FUNCTIONS.
from avalanche.training.plugins import (
    SupervisedPlugin,
    EvaluationPlugin,
    AGEMPlugin,
    SynapticIntelligencePlugin,
)

class MySupervisedTemplate(SupervisedTemplate):
    def __init__(self, parser:argparse, checkpoint=None,  device="cpu", test_dl = None,logger = None,   plugins: Optional[Sequence["SupervisedPlugin"]] = None, evaluator=default_evaluator,  eval_every=-1, **base_kwargs):
        super().__init__(
            model=parser.model,
            optimizer=parser.optimizer,
            criterion=parser.criterion,
            train_mb_size=parser.train_mb_size,
            train_epochs=parser.train_epochs,
            eval_mb_size=parser.eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
        )

        self.test_dl = test_dl
        self.logger = logger
        self.accuracy_fun = multi_label_accuracy_weighted
        self.checkpoint = checkpoint
        self.optimum = 0.0
        self.pretrined_model = parser.pretrained_model
        self.EpochClock = EpochClock(parser.epochs, self.pretrined_model)
        self.plugins.append(self.EpochClock)
        self.scheduler = plugins[0].scheduler
        self.parser = parser
        self.epoch = 0

    @property
    def mb_x(self):
        """Current mini-batch input."""
        if len(self.mbatch[1].shape) == 1:
         self.mbatch[1] = self.mbatch[1].view([-1,1])
        return self.mbatch[:-1]

    def criterion(self):
        """Loss function.""" 
        return self._criterion(self.parser, self.mb_x, self.mb_output)

    def eval_epoch(self, **kwargs):
        """Evaluation loop over the current `self.dataloader`."""
        super().eval_epoch()
    #    print(self.evaluator.get_last_metrics())
        sum = 0.0
        for self.mbatch in self.dataloader:
            self._unpack_minibatch()
            self._before_eval_iteration(**kwargs)
            self._before_eval_forward(**kwargs)
            self.mb_output = self.forward()
            mb_output = self.model.forward_and_out_to_struct(self.mbatch[:5])
            input_struct = self.model.inputs_to_struct(self.mbatch[:5])
            self._after_eval_forward(**kwargs)
            self.loss = self.criterion()
            _ , acc = self.accuracy_fun(mb_output, input_struct)
            sum += acc
            self._after_eval_iteration(**kwargs)
        sum = sum / len(self.dataloader)
        # TODO - SUPPORT ALSO Omniglot.
        self.checkpoint(self.model, self.epoch, sum, self.optimizer, self.scheduler, self.parser,0, 2)  # Updating checkpoint.
        self.epoch += 1

class MyEWC(MySupervisedTemplate):
    def __init__(
            self,
            parser: argparse,
            mode: str = "separate",
            decay_factor: Optional[float] = None,
            keep_importance_data: bool = False,
         #   eval_mb_size: int = None,
            device=None,
            plugins: Optional[List[SupervisedPlugin]] = None,
            evaluator = None,
          #  evaluator: EvaluationPlugin = default_evaluator(),
            eval_every=-1,
            **base_kwargs
    ):
        """Init.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param ewc_lambda: hyperparameter to weigh the penalty inside the total
               loss. The larger the lambda, the larger the regularization.
        :param mode: `separate` to keep a separate penalty for each previous
               experience. `onlinesum` to keep a single penalty summed over all
               previous tasks. `onlineweightedsum` to keep a single penalty
               summed with a decay factor over all previous tasks.
        :param decay_factor: used only if mode is `onlineweightedsum`.
               It specify the decay term of the importance matrix.
        :param keep_importance_data: if True, keep in memory both parameter
                values and importances for all previous task, for all modes.
                If False, keep only last parameter values and importances.
                If mode is `separate`, the value of `keep_importance_data` is
                set to be True.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param base_kwargs: any additional
            :class:`~Baselines_code.training.BaseTemplate` constructor arguments.
        """
        ewc = MyEWCPlugin(parser, mode, decay_factor, keep_importance_data,parser.model, parser.old_dataset)
        if plugins is None:
            plugins = [ewc]
        else:
            plugins.append(ewc)

        super().__init__(
            parser,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )

class MySI(MySupervisedTemplate):
    def __init__(
            self,
            parser,
            device="cpu",
            plugins: Optional[Sequence["SupervisedPlugin"]] = None,
            evaluator = None,
            eval_every=-1,
            **base_kwargs
    ):
        """Init.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param ewc_lambda: hyperparameter to weigh the penalty inside the total
               loss. The larger the lambda, the larger the regularization.
        :param mode: `separate` to keep a separate penalty for each previous
               experience. `onlinesum` to keep a single penalty summed over all
               previous tasks. `onlineweightedsum` to keep a single penalty
               summed with a decay factor over all previous tasks.
        :param decay_factor: used only if mode is `onlineweightedsum`.
               It specify the decay term of the importance matrix.
        :param keep_importance_data: if True, keep in memory both parameter
                values and importances for all previous task, for all modes.
                If False, keep only last parameter values and importances.
                If mode is `separate`, the value of `keep_importance_data` is
                set to be True.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param base_kwargs: any additional
            :class:`~Baselines_code.training.BaseTemplate` constructor arguments.
        """
        if plugins is None:
            plugins = []

        # This implementation relies on the S.I. Plugin, which contains the
        # entire implementation of the strategy!
        plugins.append(MySIPlugin( parser = parser, eps = parser.si_eps ))

        super().__init__(
            parser,                             
            device = device,
            plugins = plugins,
            evaluator = evaluator,
            eval_every = eval_every,
            **base_kwargs
        )

class Mylwf(MySupervisedTemplate):
    def __init__(
            self,
            parser,
            train_mb_size: int = 1,
            train_epochs: int = 1,
            eval_mb_size: int = None,
            device=None,
            plugins: Optional[List[SupervisedPlugin]] = None,
            evaluator: EvaluationPlugin = default_evaluator,
            eval_every=-1,
            **base_kwargs
    ):
        """Init.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param alpha: distillation hyperparameter. It can be either a float
                number or a list containing alpha for each experience.
        :param temperature: softmax temperature for distillation
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param base_kwargs: any additional
            :class:`~Baselines_code.training.BaseTemplate` constructor arguments.
        """

        model = parser.model
        optimizer = parser.optimizer
        criterion = parser.criterion
        if parser.pretrained_model:
            prev_model = model
        else:
         prev_model = None
        lwf = MyLwFPlugin(parser, prev_model)
        if plugins is None:
            plugins = [lwf]
        else:
            plugins.append(lwf)

        super().__init__(
            parser = parser,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )

class MyLFL(MySupervisedTemplate):
    """Less Forgetful Learning strategy.

    See LFL plugin for details.
    Refer Paper: https://arxiv.org/pdf/1607.00122.pdf
    This strategy does not use task identities.
    """

    def __init__(
            self,
            parser: argparse,
            device=None,
            plugins: Optional[List[SupervisedPlugin]] = None,
            evaluator=None,
            #  evaluator: EvaluationPlugin = default_evaluator(),
            eval_every=-1,
            **base_kwargs
    ):
        """Init.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param lambda_e: euclidean loss hyper parameter. It can be either a
                float number or a list containing lambda_e for each experience.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
            :class:`~Baselines_code.training.BaseTemplate` constructor arguments.
        """

        lfl = MyLFLPlugin(parser.lfl_lambda, parser.model)
        if plugins is None:
            plugins = [lfl]
        else:
            plugins.append(lfl)

        super().__init__(
            parser,
            device=device,
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

    def __init__(
            self,
            parser: argparse,
            device=None,
            plugins: Optional[List[SupervisedPlugin]] = None,
            evaluator=None,
            #  evaluator: EvaluationPlugin = default_evaluator(),
            eval_every=-1,
            **base_kwargs
    ):
        """Init.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param lambda_e: euclidean loss hyper parameter. It can be either a
                float number or a list containing lambda_e for each experience.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
            :class:`~Baselines_code.training.BaseTemplate` constructor arguments.
        """

        MAS = MyMASPlugin(parser)
        if plugins is None:
            plugins = [MAS]
        else:
            plugins.append(MAS)

        super().__init__(
            parser,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )

class MyRWALK(MySupervisedTemplate):
    def __init__(
            self,
            parser: argparse,
            device=None,
            plugins: Optional[List[SupervisedPlugin]] = None,
            evaluator=None,
            #  evaluator: EvaluationPlugin = default_evaluator(),
            eval_every=-1,
            **base_kwargs
    ):
        """Init.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param lambda_e: euclidean loss hyper parameter. It can be either a
                float number or a list containing lambda_e for each experience.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
            :class:`~Baselines_code.training.BaseTemplate` constructor arguments.
        """

        RWalk = RWalkPlugin(parser)
        if plugins is None:
            plugins = [RWalk]
        else:
            plugins.append(RWalk)

        super().__init__(
            parser,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )

class AGEM(MySupervisedTemplate):
    """Average Gradient Episodic Memory (A-GEM) strategy.

    See AGEM plugin for details.
    This strategy does not use task identities.
    """

    def __init__(
        self,
        parser,
        old_data = None,
        patterns_per_exp: int = 1000,
        sample_size: int = 64,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = None,
        eval_every=-1,
        **base_kwargs
    ):
        """Init.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param patterns_per_exp: number of patterns per experience in the memory
        :param sample_size: number of patterns in memory sample when computing
            reference gradient.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param base_kwargs: any additional
            :class:`~avalanche.training.BaseTemplate` constructor arguments.
        """

        agem = MyAGEMPlugin(parser,old_data, patterns_per_exp, sample_size)
        if plugins is None:
            plugins = [agem]
        else:
            plugins.append(agem)

        super().__init__(                                                                            
            parser = parser,
            device = device,
            plugins = plugins,
            evaluator = evaluator,
            eval_every = eval_every,
            **base_kwargs
        )

class GEM(MySupervisedTemplate):
    """Average Gradient Episodic Memory (A-GEM) strategy.

    See AGEM plugin for details.
    This strategy does not use task identities.
    """

    def __init__(
        self,
        parser,
        old_data = None,
        checkpoint = None,
        sample_size: int = 64,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = None,
        eval_every=-1,
        **base_kwargs
    ):
        """Init.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param patterns_per_exp: number of patterns per experience in the memory
        :param sample_size: number of patterns in memory sample when computing
            reference gradient.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param base_kwargs: any additional
            :class:`~avalanche.training.BaseTemplate` constructor arguments.
        """

        gem = MyGEMPlugin(parser,old_data, parser.patterns_per_exp, sample_size)
        if plugins is None:
            plugins = [gem]
        else:
            plugins.append(gem)

        super().__init__(
            parser = parser,
            device = device,
            checkpoint=checkpoint,
            plugins = plugins,
            evaluator = evaluator,
            eval_every = eval_every,
            **base_kwargs
        )

class MyIL(MySupervisedTemplate):
    """Average Gradient Episodic Memory (A-GEM) strategy.

    See AGEM plugin for details.
    This strategy does not use task identities.
    """

    def __init__(
        self,
        parser,
        old_data = None,
        checkpoint = None,
        patterns_per_exp: int = 100000,
        sample_size: int = 64,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = None,
        eval_every=-1,
        **base_kwargs
    ):
        """Init.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param patterns_per_exp: number of patterns per experience in the memory
        :param sample_size: number of patterns in memory sample when computing
            reference gradient.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param base_kwargs: any additional
            :class:`~avalanche.training.BaseTemplate` constructor arguments.
        """

        gem = IL(parser,old_data, patterns_per_exp, sample_size)
        if plugins is None:
            plugins = [gem]
        else:
            plugins.append(gem)

        super().__init__(
            parser = parser,
            device = device,
            checkpoint=checkpoint,
            plugins = plugins,
            evaluator = evaluator,
            eval_every = eval_every,
            **base_kwargs
        )

class Naive_with_freezing(MySupervisedTemplate):
    """Naive finetuning.

    The simplest (and least effective) Continual Learning strategy. Naive just
    incrementally fine tunes a single model without employing any method
    to contrast the catastrophic forgetting of previous knowledge.
    This strategy does not use task identities.

    Naive is easy to set up and its results are commonly used to show the worst
    performing baseline.
    """

    def __init__(
        self,
        model: Module,
        optimizers: List[Optimizer],
        criterion=CrossEntropyLoss(),
        train_mb_size: int = 1,
        train_epochs: list[int] = [1],
        eval_mb_size: Optional[int] = None,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = None,
        eval_every=-1,
        **base_kwargs
    ):
        """
        Creates an instance of the Naive strategy.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param base_kwargs: any additional
            :class:`~Baselines_code.training.BaseTemplate` constructor arguments.
        """
        self.optimizers = optimizers
        self.epochs_list = train_epochs
        optimizer = optimizers[0]
        train_epoch = train_epochs[0]
        super().__init__(
            model = model,
            optimizer = optimizer,
            criterion = criterion,
            train_mb_size=train_mb_size,
            train_epochs = train_epoch,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )
        
    def _after_training_exp(self, **kwargs):
        """
        Switching the optimizer, number of parameters.
        Args:
            **kwargs:

        Returns:

        """
        super(Naive_with_freezing, self)._after_training_exp()
        nexperiences_trained_so_far = self.clock.train_exp_counter
        if nexperiences_trained_so_far  < len(self.optimizers):
          new_exp_idx= nexperiences_trained_so_far
          self.optimizer = self.optimizers[new_exp_idx]
          self.train_epochs = self.epochs_list[new_exp_idx]
          print("switch_optimizer")
        
             
