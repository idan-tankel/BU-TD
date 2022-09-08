import enum
import torch.optim as optim
import logging
import os
import numpy as np
import time
import torch
import shutil
import argparse
from Configs.Config import Config
from v26.models.DatasetInfo import DatasetInfo
from torch.utils.data import DataLoader
from torch import nn
from enum import Enum




class save_details_class():
    """
    Enum class to define the save details.
    """
    train = 0
    test = 1
    val = 2

    def __init__(self, epoch_save_index: str, dataset_name: str):
        """
        :param opts: The options de decide the metric and the dataset to update the model according to.
        """
        self.optimum = -np.inf  # The initial optimum.
        # The metric we should save according to.
        self.epoch_save_idx = epoch_save_index
        if dataset_name == 'train':
            self.dataset_id = 0  # The dataset to save according to.
        elif dataset_name == 'test':
            self.dataset_id = 1
        else:
            self.dataset_id = 2
        if self.epoch_save_idx == 'loss':
            self.epoch_save_idx = 0
        else:
            self.epoch_save_idx = 1


def train_model(args: Config, the_datasets: list, learned_params: list, task_id: int, model) -> None:
    """
    This function wrapped the training process. It's set up the optimizer, the loss function, the scheduler.
    The main function within it is `fit` function.

    # TODO : add the option to load the model as an argument and not from the argparse.


    :param args: The model options.
    :param the_datasets: The datasets to train on
    :param learned_params: The parameters we train.
    :param task_id: The task_id
    :param model: The model we fit.
    :return:
    """
    # TODO - separate this part from thr train_model.
    if args.Training.optimizer == 'SGD':
        optimizer = optim.SGD(learned_params, lr=args.Training.lr,
                              momentum=args.Training.momentum, weight_decay=args.Training.weight_decay)
        if args.Training.cycle_lr:
            scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.Training.lr, max_lr=args.Training.max_lr,
                                                    step_size_up=args.nbatches_train // 2, step_size_down=None,
                                                    mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle',
                                                    cycle_momentum=True, base_momentum=0.8, max_momentum=0.9,
                                                    last_epoch=-1)
    else:
        optimizer = optim.Adam(
            learned_params, lr=args.Training.lr, weight_decay=args.Training.weight_decay)
        if args.Training.cycle_lr:
            scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.Training.lr, max_lr=args.Training.max_lr,
                                                    step_size_up=args.nbatches_train // 2, step_size_down=None,
                                                    mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle',
                                                    cycle_momentum=False, last_epoch=-1)

    args.Training.optimizer = optimizer
    if not args.Training.cycle_lr:
        def lmbda(epoch): return args.learning_rates_mult[epoch]
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)
    args.scheduler = scheduler
    fit(args, the_datasets, model=model,optimizer=optimizer, scheduler=scheduler)
    # TODO find what the task_id can be (right/left)


def create_optimizer_and_sched(opts: Config) -> tuple:
    """
    :param opts: The model options (specificly training options subsection).
    :return: optimizer, scheduler according to the options.
    """
    if opts.SGD:
        optimizer = optim.SGD(learned_params, lr=opts.initial_lr,
                              momentum=arg.momentum, weight_decay=opts.weight_decay)
        if opts.cycle_lr:
            scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=opts.base_lr, max_lr=opts.max_lr,
                                                    step_size_up=nbatches_train // 2, step_size_down=None,
                                                    mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle',
                                                    cycle_momentum=True, base_momentum=0.8, max_momentum=0.9,
                                                    last_epoch=-1)
    else:
        optimizer = optim.Adam(
            learned_params, lr=opts.base_lr, weight_decay=opts.wd)
        if opts.cycle_lr:
            scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=opts.base_lr, max_lr=opts.max_lr,
                                                    step_size_up=opts.nbatches_train // 2, step_size_down=None,
                                                    mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle',
                                                    cycle_momentum=False, last_epoch=-1)

    if not opts.cycle_lr:
        def lmbda(epoch): return opts.learning_rates_mult[epoch]
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)
    return optimizer, scheduler


def train_step(opts: argparse, inputs: list[torch]) -> tuple:
    """
    Making a train step in which we update the model.
    :param opts: The model options.
    :param inputs: The model inputs.
    :return: The loss and output of the batch.
    """
    opts.model.train()  # Move the model into the train mode.
    outs = opts.model(inputs)  # Compute the model output.
    loss = opts.loss_fun(opts.model, inputs, outs)  # Compute the loss.
    opts.optimizer.zero_grad()  # Reset the optimizer.
    loss.backward()  # Do a backward pass.
    opts.optimizer.step()  # Update the model.
    if type(opts.scheduler) in [optim.lr_scheduler.CyclicLR,
                                optim.lr_scheduler.OneCycleLR]:  # Make a scedular step if needed.
        opts.scheduler.step()
    return loss, outs  # Return the loss and the output.


def test_step(opts: argparse, inputs: list[torch]) -> tuple:
    """
    Making a test step in which we don't update the model.
    :param opts: model options.
    :param inputs: The inputs
    :return: The loss and the model output.
    """
    opts.model.eval()  # Move the model to evaluation mode in order to not change the running statistics of the batch layers.
    with torch.no_grad():  # Don't need to compute grads.
        outs = opts.model(inputs)  # Compute the model outputs.
        loss = opts.loss_fun(opts.model, inputs, outs)  # Compute the loss.
    return loss, outs  # Return the loss and the output.


'''
def from_network_transpose(samples, outs):
    if normalize_image:
        samples.image += mean_image
    samples.image = samples.image.transpose(0, 2, 3, 1)
    samples.seg = samples.seg.transpose(0, 2, 3, 1)
    if model_opts.use_td_loss:
        outs.td_head = outs.td_head.transpose(0, 2, 3, 1)
    return samples, outs
'''


def save_model_and_md(logger: logging, model_fname: str, metadata: dict, epoch: int, opts: argparse,model: nn.Module) -> None:
    """
    Saving the model weights and the metadata(e.g. running statistics of the batch norm layers)
    :param logger: The logger we write into.
    :param model_fname: The path to save in.
    :param metadata: The metadata we desire to save.
    :param epoch: The epoch_id.
    :param opts: The model options.
    :return:
    """
    tmp_model_fname = model_fname + '.tmp'  # Save into temporal directory,
    # Send a message into the logger.
    logger.info('Saving model to %s' % model_fname)
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opts.Training.optimizer.state_dict(),
                'scheduler_state_dict': opts.scheduler.state_dict(), 'metadata': metadata, },
               tmp_model_fname)  # Saving the metadata.
    os.rename(tmp_model_fname, model_fname)  # Rename to the desired name.
    logger.info('Saved model')


def load_model(opts: argparse, model_path: str, model_latest_fname: str,model: nn.Module, gpu=None,
               load_optimizer_and_schedular: bool = False) -> dict:
    """
    Loads and returns the model as a dictionary.
    :param opts: The model opts.
    :param model_path: The model path
    :param model_latest_fname: The model id in the folder.
    :param model: The model object
    :param gpu:
    :param load_optimizer_and_schedular: Whether to load optimizer and schedular.
    :return: The model from checkpoint.
    """
    if gpu is None:
        model_path = os.path.join(model_path, model_latest_fname)
        # Loading the weights and the metadata.
        checkpoint = torch.load(model_path)
    else:
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(gpu)
        checkpoint = torch.load(model_path, map_location=loc)
    # Loading the epoch_id, the optimum and the data.
    model.load_state_dict(checkpoint['model_state_dict'])
    if load_optimizer_and_schedular:
        # Load the optimizer state.
        opts.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Load the schedular state.
        opts.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint


def fit(opts: argparse, the_datasets: list[DatasetInfo],  model:nn.Module,optimizer,scheduler) -> None:
    """
    Fitting the model.
    iterate over the datasets and train (or test) them
    :param opts: The model options. Saved as Config object
    :param the_datasets: The train, test datasets. List of DatasetInfo objects
    :param model: The model object. An extention of nn.Module.
    :param optimizer: The optimizer object.
    :param scheduler: The scheduler object.
    """
    logger = opts.logger
    #  if opts.first_node:
    logger.info('train_opts: %s', str(opts))
    # Getting the number of epoch we desire to train for,
    nb_epochs = opts.Training.epochs
    model_dir = opts.model_dir  # Getting the model directory.
    model_ext = '.pt'
    model_basename = 'model'
    model_latest_fname = model_basename + '_latest' + model_ext
    model_latest_fname = os.path.join(model_dir, model_latest_fname)
    # Contains the optimum.
    save_details = save_details_class(
        epoch_save_index=opts.Training.epoch_save_idx, dataset_name=opts.Training.dataset_to_save)
    last_epoch = -1
    # If we want to continue an old training.
    if opts.Training.load_model_if_exists:
        if os.path.exists(model_latest_fname):
            logger.info('Loading model: %s' % model_latest_fname)
            checkpoint = load_model(opts, model_latest_fname, opts.gpu)
            metadata = checkpoint['metadata']
            for the_dataset in the_datasets:
                the_dataset.measurements.results = metadata[the_dataset.name]
            optimum = metadata['optimum']
            last_epoch = metadata['epoch']
            if opts.distributed:
                # synchronize point so all distributed processes would have the same weights
                import torch.distributed as dist
                dist.barrier()
            logger.info('restored model with optimum %f' % optimum)
            logger.info('continuing from epoch: %d' % (last_epoch + 2))

    st_epoch = last_epoch + 1
    end_epoch = nb_epochs

    # Training for end_Epoch - st_epoch epochs.
    for epoch in range(st_epoch, end_epoch):
        #   if opts.first_node:
        # The current lr.
        logger.info('Epoch {} learning rate: {}'.format(
            epoch + 1, optimizer.param_groups[0]['lr']))
        # Training/testing the model by the datasets.
        for the_dataset in the_datasets:
            the_dataset.do_epoch(opts=opts, epoch=epoch,
                                 number_of_epochs=nb_epochs, model=model)
        # logger info done the epoch.
        opts.logger.info('Epoch {} done'.format(epoch + 1))
        #
        # Storing the running stats to avoid forgetting.
        # store_running_stats(opts.model, task)
        # Adding to the logger.
        # logger.info('epoch {} done storing running stats'.format(epoch + 1))
        #   print("Done storing running stats!")
        #
        for the_dataset in the_datasets:  # Printing the loss, accuracy per dataset.
            logger.info('Epoch {}, {} {}'.format(
                epoch + 1, the_dataset.name, the_dataset.measurements.print_epoch()))
        if opts.scheduler is not None:  # Scheduler step.
            if not type(opts.scheduler) in [optim.lr_scheduler.CyclicLR, optim.lr_scheduler.OneCycleLR]:
                if epoch < nb_epochs - 1:
                    # learning rate scheduler
                    # Scheduler step in case we reduce lr by loss.
                    if type(opts.scheduler) is optim.lr_scheduler.ReduceLROnPlateau:
                        the_val_dataset = the_datasets[-1]
                        val_loss = the_val_dataset.measurements.results[-1, 0]
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()

        if opts.Training.save_model:  # Saving the model.
            optimum = save_details.optimum  # Getting the old optimum.
            save_by_dataset = the_datasets[
                save_details.dataset_id]  # Getting the dataset we update the model according to.
            measurements = np.array(save_by_dataset.measurements.results)
            # Flag telling whether we overcome the old optimum.
            new_optimum = False
            # Getting the possible new optimum.
            epoch_save_value = measurements[epoch, save_details.epoch_save_idx]
            if epoch_save_value > optimum:  # If we overcome the old optimum.
                optimum = epoch_save_value  # Update the local optimum.
                new_optimum = True  # Change the flag
                # Adding to the logger.
                logger.info('New optimum: %f' % optimum)
                save_details.optimum = optimum  # Update the optimum.
            metadata = dict()  # metadata dictionary.
            metadata['epoch'] = epoch
            for the_dataset in the_datasets:  # Storing the loss, accuracy.
                measurements = np.array(the_dataset.measurements.results)
                metadata[the_dataset.name] = measurements
            metadata['optimum'] = optimum  # Storing the new optimum.
            model_latest_fname = model_basename + '_latest' + model_ext
            model_latest_fname = os.path.join(model_dir, model_latest_fname)
            save_model_and_md(logger, model_latest_fname, metadata, epoch,
                              opts,model=model)  # Storing the metadata in the model_latest.
            # If we have a new optimum, we store in an additional model to avoid overriding.
            if new_optimum:
                model_fname = model_basename + '%d' % (epoch + 1) + model_ext
                model_fname = os.path.join(model_dir, model_fname)
                shutil.copyfile(model_latest_fname, model_fname)
                logger.info('Saved model to %s' % model_fname)

    logger.info('Done fit')
