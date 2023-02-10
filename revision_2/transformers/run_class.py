import logging
import os
import time

import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import SimpleProfiler
from pytorch_lightning.strategies import DDPStrategy
from torch import Tensor

from common.common_functions import get_saved_model_directory, get_checkpoint_location
from common.trainingOptions import TrainingOptions
from transformers.Utils.TrainingData import TrainData
from transformers.configs.config import Config, TrainingOptionsConfig
from transformers.models.EpochProgressBar import EpochProgressBar
from transformers.models.Transformers import MultiHeadsOutputVit
from transformers.objectss_of_tasks.AllObjectTasks import AllObjectTasks
from transformers.objectss_of_tasks.ObjectTaskInterface import ObjectTaskInterface
from transformers.vit_models_only_forward.vitModelInterface import VitModelInterface
from utils.read_config import read_config, is_private


# from line_profiler_pycharm import profile


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# mpl.use('TkAgg')
# matplotlib.use('TkAgg')
# plt.switch_backend('TkAgg')


def logger_init(config: Config):
    logging.basicConfig(format=('%(asctime)s ' + '%(message)s'), level=logging.INFO,
                        filename=os.path.join(config.running_specification.running.location_to_save_model, 'log.txt'))

    logger = logging.getLogger(__name__)

    return logger


def get_object_task(config) -> [ObjectTaskInterface, VitModelInterface]:
    task_object: ObjectTaskInterface = AllObjectTasks.get_wanted_object_task(
        task_name=config.running_specification.task_name,
        task_dataset=config.datasets_specs.chosen_dataset.dataset_name)
    config.datasets_specs.chosen_dataset.measurements_function = task_object.measurements
    config.datasets_specs.chosen_dataset.dataset_function = task_object.dataset_function

    task_object.put_model(model_name=config.running_specification.model.transformer_model_name,
                          model_implementation=config.running_specification.model.transformer_model_implementation)

    return task_object


def get_model(object_task: ObjectTaskInterface, config: Config, logger: logging.Logger,
              train_data: TrainData) -> MultiHeadsOutputVit:
    mean: Tensor = torch.tensor(config.datasets_specs.chosen_dataset.mean)
    std: Tensor = torch.tensor(config.datasets_specs.chosen_dataset.std)
    number_of_tasks = config.datasets_specs.chosen_dataset.nclasses_existence

    # model_path_to_load: str = get_saved_model_location(config)
    model_path_to_load: str = get_checkpoint_location(config)

    # Add loading model
    if not config.saved_model_specifications.is_force_create_Model and (
            config.running_specification.running.is_load_model and os.path.exists(model_path_to_load)):
        logger.info("Loading model from: " + model_path_to_load)
        # model = torch.load(model_path_to_load)  # TODO - take the model name from config
        model: MultiHeadsOutputVit = MultiHeadsOutputVit.load_from_checkpoint(model_path_to_load)
    else:
        model: MultiHeadsOutputVit = \
            MultiHeadsOutputVit(config.running_specification.model.transformer_model_implementation,
                                config.running_specification.model.transformer_model_input_shape,
                                number_of_classes=config.datasets_specs.chosen_dataset.nclasses_existence,
                                number_of_heads=number_of_tasks, mean=mean, std=std, config=config,
                                object_task=object_task, train_data=train_data,
                                batch_size=config.running_specification.running.batch_size,
                                num_instructions=object_task.get_number_of_instructions())

    return model


def main():
    config: Config = init_config()
    object_task = get_object_task(config=config)
    logger: logging.Logger = logger_init(config)

    logger.info('Data set name: ' + config.datasets_specs.chosen_dataset.dataset_name)
    logger.info('Training dataset size: ' + str(config.running_specification.running.nsamples_train))

    train_data: TrainData = TrainData()

    model: MultiHeadsOutputVit = get_model(object_task, config, logger, train_data)
    # trainer.tune(model)  # TODO - add it when it will be supported for DDP by pytorch lightning
    # Pretrain the model
    pre_trainer = init_trainer(num_epochs=config.running_specification.model.training_options_config.num_epochs_pretrain,
                              logger=logger, train_data=train_data, is_pretrain=True)

    train_data.is_pretrain = True
    pre_trainer.fit(model)
    train_data.is_pretrain = False

    trainer = init_trainer(num_epochs=config.running_specification.model.training_options_config.num_epochs,
                           model_dir=os.path.join(get_saved_model_directory(config)),
                           logger=logger, train_data=train_data)
    trainer.fit(model)

    trainer.fit(model)

    trainer.test(model)

    logger.info("Training finished")
    exit(0)


def init_trainer(train_data: TrainData, logger: logging.Logger, num_epochs: int = 20,
                 model_dir: str = None, is_pretrain: bool = False) -> Trainer:
    strategy = DDPStrategy(find_unused_parameters=True)
    progress_bar: TQDMProgressBar = EpochProgressBar(train_data=train_data, is_pretrain=is_pretrain)
    if is_pretrain:
        trainer = Trainer(accelerator="gpu", strategy=strategy, devices=-1, max_epochs=num_epochs,
                          auto_lr_find=True,
                          accumulate_grad_batches=2, auto_scale_batch_size='binsearch',
                          callbacks=[progress_bar], auto_select_gpus=True,
                          reload_dataloaders_every_n_epochs=2)
    else:
        logger.info('model_dir: ' + model_dir)
        wandb_logger = WandbLogger()
        check_point_dir = os.path.join(model_dir, 'checkpoints', time.asctime().replace(" ", '_').replace(":", "_"))
        checkpoint_callback = ModelCheckpoint(dirpath=check_point_dir, monitor="val_acc", mode="max", save_top_k=10,
                                              filename='best', auto_insert_metric_name=False)
        if not os.path.isdir(check_point_dir):
            os.makedirs(check_point_dir)
        logger.info('check_point_dir: ' + check_point_dir)

        profiler = SimpleProfiler(dirpath=model_dir, filename="profile")
        trainer = Trainer(accelerator="gpu", strategy=strategy, devices=-1, max_epochs=num_epochs,
                          default_root_dir=model_dir, auto_lr_find=True, profiler=profiler, logger=wandb_logger,
                          accumulate_grad_batches=2, auto_scale_batch_size='binsearch',
                          callbacks=[progress_bar, checkpoint_callback], auto_select_gpus=True,
                          reload_dataloaders_every_n_epochs=2)
    return trainer


def check_if_test_code(config: Config):
    if config.running_specification.running.is_testing_code:
        config.running_specification.running.nsamples_train = config.running_specification.running.nsamples_train_quick
        config.running_specification.running.nsamples_test = config.running_specification.running.nsamples_test_quick
        config.running_specification.running.nsamples_val = config.running_specification.running.nsamples_val_quick


def init_config() -> Config:
    config: Config = read_config(os.path.join('configs', 'config.yaml'), Config)
    if not os.name == 'nt':
        wandb.init(project=config.project_name + "_" + config.running_specification.task_name, mode='online')

    for curr_attr in config.datasets_specs.__dir__():
        if not is_private(curr_attr) and getattr(config.datasets_specs,
                                                 curr_attr).dataset_name == config.running_specification.model.dataset_to_use:
            config.datasets_specs.chosen_dataset = getattr(config.datasets_specs, curr_attr)
            config.datasets_specs.chosen_dataset.base_samples_dir = os.path.join(
                *config.datasets_specs.chosen_dataset.base_samples_dir.replace("\\", "/").split("/"))
            config.datasets_specs.chosen_dataset.chosen_dataset_full_path = os.path.join(os.path.dirname(os.getcwd()),
                                                                                         config.datasets_specs.chosen_dataset.dataset_name,
                                                                                         config.datasets_specs.chosen_dataset.base_samples_dir)
            config.running_specification.running.inshape = config.datasets_specs.chosen_dataset.inshape
            break

    if not config.growing_data_sets.is_growing_data_sets:
        check_if_test_code(config)

    return config


if __name__ == '__main__':
    main()
