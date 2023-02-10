import logging
import os

import numpy as np
import torch
from typing import Tuple

from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

from common.common_classes import DataInputs
from common.onTheRunInfo import OnTheRunInfo
from transformers.configs.config import Config

logging.basicConfig(format=('%(asctime)s ' + '%(message)s'), level=logging.INFO)

logger = logging.getLogger(__name__)


#######################################
#    Train functions
#######################################
def get_multi_gpu_learning_rate(learning_rates_mult, num_gpus, scale_batch_size, ubs):
    # In pytorch gradients are summed across multi-GPUs (and not averaged) so
    # there is no need to change the learning rate when changing from a single GPU to multiple GPUs.
    # However, when increasing the batch size (not because of multi-GPU, i.e. when scale_batch_size>1),
    # we need to increase the learning rate as usual
    clr = False
    if clr:
        learning_rates_mult *= scale_batch_size
    else:
        if ubs > 1:
            warmup_epochs = 5
            initial_lr = np.linspace(learning_rates_mult[0] / num_gpus, scale_batch_size * learning_rates_mult[0],
                                     warmup_epochs)
            learning_rates_mult = np.concatenate((initial_lr, scale_batch_size * learning_rates_mult))
    return learning_rates_mult


def save_model_and_md(model_fname, metadata, epoch, opts):
    tmp_model_fname = model_fname + '.tmp'
    logger.info('Saving model to %s' % model_fname)
    torch.save({
        'epoch': epoch,
        'model_state_dict': opts.model.state_dict(),
        'optimizer_state_dict': opts.optimizer.state_dict(),
        'scheduler_state_dict': opts.scheduler.state_dict(),
        'metadata': metadata,
    }, tmp_model_fname)
    os.rename(tmp_model_fname, model_fname)
    logger.info('Saved model')


def load_model(opts, model_latest_fname, gpu=None):
    if gpu is None:
        checkpoint = torch.load(model_latest_fname)
    else:
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(gpu)
        checkpoint = torch.load(model_latest_fname, map_location=loc)
    # checkpoint = torch.load(model_latest_fname)
    opts.model.load_state_dict(checkpoint['model_state_dict'])
    opts.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    opts.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint


def get_inputs_from_list(inputs) -> DataInputs:
    inputs: DataInputs = DataInputs(inputs)
    return inputs


def train_step(inputs, training_options, number_classes: int):
    training_options.model.train()

    outs = training_options.model(inputs.images)
    loss = training_options.loss_fun(torch.squeeze(inputs.label_existence, 1), outs, number_classes)

    training_options.optimizer.zero_grad()
    loss.backward()
    training_options.optimizer.step()
    return loss, outs


def test_step(inputs, training_options, number_classes: int):
    training_options.model.eval()
    with torch.no_grad():
        outs = training_options.model(inputs.images)
        loss = training_options.loss_fun(torch.squeeze(inputs.label_existence, 1), outs, number_classes)
    return loss, outs


def set_datasets_measurements(datasets, measurements_class, model_opts, model):
    for the_dataset in datasets:
        the_dataset.create_measurement(measurements_class, model_opts, model)


def get_saved_model_location(config: Config) -> str:
    directory_of_chosen_transformer_model: str = get_saved_model_directory(config)
    model_path_to_load: str = os.path.join(directory_of_chosen_transformer_model,
                                           config.saved_model_specifications.best_model_saved_name +
                                           config.saved_model_specifications.file_extension)
    return model_path_to_load


def get_checkpoint_location(config: Config) -> str:
    directory_of_chosen_transformer_model: str = get_saved_model_directory(config)
    model_path_to_load: str = os.path.join(directory_of_chosen_transformer_model,
                                           config.saved_model_specifications.checkpoint +
                                           config.saved_model_specifications.checkpoint_extension)
    return model_path_to_load


def get_saved_model_directory(config: Config) -> str:
    directory_of_results: str = os.path.join(os.getcwd(), config.saved_model_specifications.location_to_save_model)
    directory_database_output: str = os.path.join(directory_of_results,
                                                  config.running_specification.model.dataset_to_use)
    directory_of_chosen_transformer_model: str = \
        os.path.join(directory_database_output, config.running_specification.model.transformer_model_implementation)

    return directory_of_chosen_transformer_model


def get_saved_model_hops_location(config: Config, next_hop_index: int) -> str:
    directory_of_hops_training = get_directory_of_hops_training(config)
    model_path_to_load: str = os.path.join(directory_of_hops_training,
                                           config.growing_data_sets.name_save_models + str(
                                               next_hop_index - 1) + config.saved_model_specifications.file_extension)
    return model_path_to_load


def get_directory_of_hops_training(config: Config) -> str:
    directory_of_chosen_transformer_model: str = get_saved_model_directory(config)
    directory_of_hops_training: str = os.path.join(directory_of_chosen_transformer_model,
                                                   config.growing_data_sets.location_to_save_output)
    return directory_of_hops_training


def create_dir_of_model(create_parent_dir: str):
    if not os.path.isdir(os.path.dirname(create_parent_dir)):
        os.makedirs(os.path.dirname(create_parent_dir))


def exists_larger_dataset_size_mean_std(config):
    return os.path.isfile(os.path.join(name_file_std_mean(config)))  # TODO - find if exists larger


def load_mean_std_from_file(config, logger: logging.Logger) -> Tuple[float, float]:
    logger.info('Loading mean and std from file')
    dict_std_mean = torch.load(name_file_std_mean(config))
    return dict_std_mean['mean'], dict_std_mean['std']


def get_mean_and_std(dataloader: DataLoader, inputs_to_struct, logger: logging.Logger, config: Config):
    if config.datasets_specs.chosen_dataset.mean is not None and config.datasets_specs.chosen_dataset.std is not None:
        mean = torch.tensor(config.datasets_specs.chosen_dataset.mean)
        std = torch.tensor(config.datasets_specs.chosen_dataset.std)
        for inputs in dataloader:
            samples = inputs_to_struct(inputs)
            number_of_tasks = samples.seg.shape[2]  # TODO - maybe -1
            return mean, std, number_of_tasks
    else:
        logger.info('Computing mean and std of dataset')
        channels_sum, channels_squared_sum, num_batches, number_of_tasks = 0, 0, 0, 0
        for inputs in dataloader:
            samples = inputs_to_struct(inputs)
            # Mean over batch, height and width, but not over the channels
            images = samples.image
            channels_sum += torch.mean(images, dim=[0, 2, 3])
            channels_squared_sum += torch.mean(images ** 2, dim=[0, 2, 3])
            number_of_tasks = samples.seg.shape[2]  # TODO - maybe -1

            num_batches += 1

        mean = channels_sum / num_batches

        # std = sqrt(E[X^2] - (E[X])^2)
        std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
        logger.info('Computed mean and std of dataset: mean: ' + str(mean) + ' std: ' + str(std))

        save_mean_std(mean, std, config, logger)
        return mean, std, number_of_tasks


def tocpu(inputs):
    return [inp.cpu() if inp is not None else None for inp in inputs]


def tonp(inputs):
    inputs = tocpu(inputs)
    return [inp.numpy() if inp is not None else None for inp in inputs]


def detach_tonp(outs):
    outs = [out.detach() if out is not None else None for out in outs]
    outs = tonp(outs)
    return outs


def name_file_std_mean(config):
    return os.path.join(get_saved_model_directory(config=config),
                        str(config.running_specification.running.nsamples_train) + '_mean_std.pkl')


def save_mean_std(mean: float, std: float, config: Config, logger: logging.Logger):
    directory: str = get_saved_model_directory(config=config)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save({'mean': mean, 'std': std}, name_file_std_mean(config))
    logger.info('Saved mean and std to file: ' + name_file_std_mean(config))


def get_the_output_right_of(labels, number_classes, outs):
    preds = torch.zeros(labels.shape[0], number_classes,
                        dtype=torch.int)
    for k in range(number_classes):
        start_task_index = int(k * (number_classes + 1))
        end_task_index = int(start_task_index + (number_classes + 1))
        taskk_out = outs[:, start_task_index:end_task_index]
        predsk = torch.argmax(taskk_out, axis=1)  # TODO - axis should be dim?
        preds[:, k] = predsk
    return preds


def clear_fig(fig):
    fig.clf()
    fig.tight_layout()


def pause_image(fig=None):
    plt.draw()
    plt.show(block=False)
    if fig == None:
        fig = plt.gcf()

    #    fig.canvas.manager.window.activateWindow()
    #    fig.canvas.manager.window.raise_()
    fig.waitforbuttonpress()


def get_loss_function(weights_losses=None):
    if weights_losses is None:
        return nn.CrossEntropyLoss(reduction='none').to(OnTheRunInfo.dev)
    return nn.CrossEntropyLoss(reduction='none', weight=weights_losses).to(OnTheRunInfo.dev)
