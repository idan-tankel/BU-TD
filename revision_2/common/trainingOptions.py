import sys

import numpy as np
import torch
from torch import optim

from common.common_functions import get_multi_gpu_learning_rate
from transformers.configs.config import TrainingOptionsConfig
from transformers.objectss_of_tasks.ObjectTaskInterface import ObjectTaskInterface


class TrainingOptions:
    scheduler = None
    ubs = None
    num_gpus_per_node = None
    learning_rates_mult = None
    optimizer = None
    distributed: bool = False
    loss_fun = None

    def __init__(self, training_options_config: TrainingOptionsConfig, model, object_task: ObjectTaskInterface):
        self.model = model
        self.training_options_config = training_options_config
        self.init_optimizer(model)
        self.init_scheduler()
        self.distributed = training_options_config.distributed
        self.task_accuracy = object_task.accuracy

    def init_scheduler(self):
        self.num_gpus_per_node = torch.cuda.device_count()
        self.ubs = self.training_options_config.scale_batch_size * self.num_gpus_per_node
        self.init_learning_rates_mult(self.training_options_config)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer,
                                                     lr_lambda=lambda epoch: self.learning_rates_mult[epoch])

    def init_learning_rates_mult(self, training_options_config):
        self.learning_rates_mult = np.ones(training_options_config.num_epochs)
        self.learning_rates_mult = get_multi_gpu_learning_rate(self.learning_rates_mult,
                                                               self.num_gpus_per_node,
                                                               training_options_config.scale_batch_size,
                                                               self.ubs)
        if training_options_config.checkpoints_per_epoch > 1:
            self.learning_rates_mult = np.repeat(self.learning_rates_mult,
                                                 training_options_config.checkpoints_per_epoch)

    def init_optimizer(self, model):
        if self.training_options_config.optimizer_name == 'Adam':
            self.optimizer = optim.Adam(model.parameters(),
                                        lr=self.training_options_config.initial_lr,
                                        weight_decay=self.training_options_config.weight_decay)
        else:
            if self.training_options_config.optimizer_name == 'SGD':
                self.optimizer = optim.SGD(model.parameters(),
                                           lr=self.training_options_config.initial_lr,
                                           momentum=self.training_options_config.momentum,
                                           weight_decay=self.training_options_config.weight_decay)
        if self.optimizer is None:
            sys.exit("Optimizer not found")
