from typing import Optional, Sequence, List, Union
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer
import torch
from avalanche.training.plugins.evaluation import default_evaluator
import sys
from avalanche.training.templates.supervised import SupervisedTemplate
sys.path.append(r'/home/sverkip/data/BU-TD/yonathan/Recognicion/code/Integration_toward_CL')
sys.path.append(r'/home/sverkip/data/BU-TD/yonathan/Recognicion/code/')
from supp.data_functions import preprocess
from supp_avalanche_AI.Plugins import MyEWCPlugin, MylwfPlugin,  MyLFLPlugin
from supp_avalanche_AI.Accuracy_plugin import multi_label_accuracy_weighted
from avalanche.training.plugins import SupervisedPlugin
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


class MySupervisedTemplate(SupervisedTemplate):
    def __init__(    self,   model: Module,  optimizer: Optimizer,  criterion=CrossEntropyLoss(),checkpoint=None,train_mb_size: int = 1, train_epochs: int = 1, eval_mb_size: Optional[int] = 1,  device="cpu", test_dl = None,logger = None,   plugins: Optional[Sequence["SupervisedPlugin"]] = None, evaluator=default_evaluator,  eval_every=-1, **base_kwargs):
        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )
        self.test_dl = test_dl
        self.logger = logger
        self.accuracy_fun = multi_label_accuracy_weighted
        self.checkpoint = checkpoint
        self.optimum = 0.0
        self.epoch = 0

    @property
    def mb_x(self):
        """Current mini-batch input."""
        if len(self.mbatch[1].shape) == 1:
         self.mbatch[1] = self.mbatch[1].view([-1,1])
        return self.mbatch[:5]

    def criterion(self):
        """Loss function."""
        return self._criterion(self.mb_x, self.mb_output)

    def eval_epoch(self, **kwargs):
        """Evaluation loop over the current `self.dataloader`."""
        sum = 0.0
        for self.mbatch in self.dataloader:
            self._unpack_minibatch()
            self._before_eval_iteration(**kwargs)

            self._before_eval_forward(**kwargs)
            self.mb_output = self.forward()
            self._after_eval_forward(**kwargs)
            self.loss = self.criterion()
            acc = self.accuracy_fun(self.mb_output[0], self.mbatch[:5])[1]
            sum += acc
            self._after_eval_iteration(**kwargs)
        sum = sum / len(self.dataloader)
        if sum > self.optimum:
            self.checkpoint(self.model,self.epoch,sum)
        self.epoch += 1
    '''
    def eval_epoch(self, **kwargs):
        sum =0.0
        with torch.no_grad():
            for inputs in self.test_dl:
                inputs = preprocess(inputs)
                outs = self.model(inputs)
                acc = self.accuracy_fun(outs[0], inputs)[1]
                loss = self._criterion(inputs,outs)
       #         self.logger.log('val_loss', loss, on_step=True, on_epoch=True, logger=True)
          #      self.logger.log('val_acc', acc, on_step=True, on_epoch=True, logger=True)
                sum += acc
            print(sum/len(self.test_dl))
            return sum/len(self.test_dl)
    '''

class MyEWC(MySupervisedTemplate):
    def __init__(
            self,
            model: Module,
            optimizer: Optimizer,
            criterion,
            ewc_lambda: float,
            start_from_regulization = False,
            Ignored_params = None,
            mode: str = "separate",
            decay_factor: Optional[float] = None,
            keep_importance_data: bool = False,
            train_mb_size: int = 1,
            train_epochs: int = 1,
            eval_mb_size: int = None,
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
            :class:`~avalanche.training.BaseTemplate` constructor arguments.
        """
        ewc = MyEWCPlugin(ewc_lambda,start_from_regulization,Ignored_params, mode, decay_factor, keep_importance_data)
        if plugins is None:
            plugins = [ewc]
        else:
            plugins.append(ewc)

        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )

class MySI(MySupervisedTemplate):
    def __init__(
            self,
            model: Module,
            optimizer: Optimizer,
            criterion,
            si_lambda: Union[float, Sequence[float]],
            eps: float = 0.0000001,
            train_mb_size: int = 1,
            train_epochs: int = 1,
            eval_mb_size: int = 1,
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
            :class:`~avalanche.training.BaseTemplate` constructor arguments.
        """
        if plugins is None:
            plugins = []

        # This implementation relies on the S.I. Plugin, which contains the
        # entire implementation of the strategy!
        plugins.append(SynapticIntelligencePlugin(si_lambda=si_lambda, eps=eps))

        super(MySI, self).__init__(
            model,
            optimizer,
            criterion,
            train_mb_size,
            train_epochs,
            eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )

class Mylwf(MySupervisedTemplate):
    def __init__(
            self,
            model: Module,
            optimizer: Optimizer,
            criterion,
            alpha: Union[float, Sequence[float]],
            temperature: float,
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
            :class:`~avalanche.training.BaseTemplate` constructor arguments.
        """

        lwf = MylwfPlugin(alpha, temperature)
        if plugins is None:
            plugins = [lwf]
        else:
            plugins.append(lwf)

        super().__init__(
            model,
            optimizer,
            criterion,
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
        model: Module,
        optimizer: Optimizer,
        criterion,
        lambda_e: Union[float, Sequence[float]],
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
            :class:`~avalanche.training.BaseTemplate` constructor arguments.
        """

        lfl = MyLFLPlugin(lambda_e)
        if plugins is None:
            plugins = [lfl]
        else:
            plugins.append(lfl)

        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
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
            :class:`~avalanche.training.BaseTemplate` constructor arguments.
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
        
        