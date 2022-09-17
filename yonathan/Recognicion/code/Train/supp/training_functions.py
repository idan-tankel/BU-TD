import torch.optim as optim
import logging
import os
import numpy as np
import argparse
import time
import torch
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from supp.FlagAt import Flag, DsType
from supp.general_functions import instruct
from supp.batch_norm import store_running_stats

class DatasetInfo:
    """encapsulates a (train/test/validation) dataset with its appropriate train or test function and Measurement class"""

    def __init__(self, istrain: bool, data_set: DataLoader, nbatches: int, name: str, checkpoints_per_epoch: int = 1,
                 sampler=None) -> None:
        """
        :param istrain: Whether we should fit the dataset.
        :param data_set: The data set.
        :param nbatches: Number of batches in the data_set.
        :param name: The dataset name
        :param checkpoints_per_epoch: Number of checkpoints in the epoch.
        :param sampler: #TODO-not clear.
        """
        self.dataset = data_set
        self.nbatches = nbatches
        self.istrain = istrain
        self.checkpoints_per_epoch = checkpoints_per_epoch
        if self.istrain and checkpoints_per_epoch > 1:
            self.nbatches = self.nbatches // checkpoints_per_epoch
        if istrain:  # If we fit the data_loader we choose each step to be train_step including backward step,schedular step.
            self.batch_fun = train_step
        else:  # otherwise we just do a forward pass and don't update the model and the schedular.
            self.batch_fun = test_step
        self.name = name
        self.dataset_iter = None
        self.needinit = True
        self.sampler = sampler

    def create_measurement(self, measurements_class: type, parser: argparse, model: nn.Module) -> None:
        """
        We create measurment object to handle our matrcies.
        :param measurements_class: The measurement class should handle the desired matrices.
        :param parser: The option parser.
        :param model: The model we fit.
        """
        self.measurements = measurements_class(parser, model)

    def reset_iter(self) -> None:
        """
        :crate a dataset iterator.
        """
        self.dataset_iter = iter(self.dataset)

    def do_epoch(self, opts: argparse, epoch: int):
        """
        :param opts:The model options.
        :param epoch: The epoch_id
        :return:
        """
        opts.logger.info(self.name)
        nbatches_report = 10
        aborted = False
        self.measurements.reset()  # Reset the measurements class.
        cur_batches = 0
        if self.needinit or self.checkpoints_per_epoch == 1:
            self.reset_iter()  # Reset the data loader to iterate from the beginning.
            self.needinit = False  # The initialization is done.
            if opts.distributed and self.sampler:
                self.sampler.set_epoch(epoch)
                # TODO: when aborted save cur_batches. next, here do for loop and pass over cur_batches
                # and use train_sampler.set_epoch(epoch // checkpoints_per_epoch)
        start_time = time.time()  # Count the beginning time.
        for inputs in self.dataset_iter:  # Iterating over all dataset.
            cur_loss, outs = self.batch_fun(opts, inputs)  # compute the model outputs and the current loss.
            with torch.no_grad():
                # so that accuracies calculation will not accumulate gradients
                self.measurements.update(inputs, outs, cur_loss.item())  # update the loss and the accuracy.
            cur_batches += 1  # Update the number of batches.
            template = 'Epoch {} step {}/{} {} ({:.1f} estimated minutes/epoch)'  # Define a convenient template.
            if cur_batches % nbatches_report == 0:
                duration = time.time() - start_time  # Compute the step time.
                start_time = time.time()
                estimated_epoch_minutes = duration / 60 * self.nbatches / nbatches_report  # compute the proportion time.
                opts.logger.info(template.format(epoch + 1, cur_batches, self.nbatches, self.measurements.print_batch(),
                                                 estimated_epoch_minutes))  # Add the epoch_id, loss, accuracy, time for epoch.
            """
            if True:
                if self.istrain and cur_batches > self.nbatches:
                    aborted = True
                    break
            """
        if not aborted:
            self.needinit = True
        self.measurements.add_history(epoch)  # Update the loss and accuracy history.


def train_model(args: argparse, the_datasets: list, learned_params: list, lang_id,direction_id) -> None:
    """
    :param args: The model options.
    :param the_datasets: The datasets.
    :param learned_params: The parameters we train.
    :param task_id: The task_id
    :return:
    """
    # TODO - separate this part from thr train_model.
    if args.SGD:
        optimizer = optim.SGD(learned_params, lr=args.initial_lr, momentum=arg.momentum, weight_decay=args.weight_decay)
        if cycle_lr:
            scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.base_lr, max_lr=args.max_lr,
                                                    step_size_up=nbatches_train // 2, step_size_down=None,
                                                    mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle',
                                                    cycle_momentum=True, base_momentum=0.8, max_momentum=0.9,
                                                    last_epoch=-1)
    else:
        optimizer = optim.Adam(learned_params, lr=args.base_lr, weight_decay=args.wd)
        if args.cycle_lr:
            scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.base_lr, max_lr=args.max_lr,
                                                    step_size_up=args.nbatches_train // 2, step_size_down=None,
                                                    mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle',
                                                    cycle_momentum=False, last_epoch=-1)

    args.optimizer = optimizer
    if not args.cycle_lr:
        lmbda = lambda epoch: args.learning_rates_mult[epoch]
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)
    args.scheduler = scheduler
    fit(args, the_datasets, lang_id,direction_id)

def create_optimizer_and_sched(opts: argparse,learned_params:list) -> tuple:
    """
    :param opts: The model options.
    :return: optimizer, scheduler according to the options.
    """
    if opts.SGD:
        optimizer = optim.SGD(learned_params, lr=opts.initial_lr, momentum=arg.momentum, weight_decay=opts.weight_decay)
        if cycle_lr:
            scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=opts.base_lr, max_lr=opts.max_lr,
                                                    step_size_up=nbatches_train // 2, step_size_down=None,
                                                    mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle',
                                                    cycle_momentum=True, base_momentum=0.8, max_momentum=0.9,
                                                    last_epoch=-1)
    else:

        optimizer = optim.Adam(learned_params, lr=opts.base_lr, weight_decay=opts.wd)

        if opts.cycle_lr:
            scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=opts.base_lr, max_lr=opts.max_lr,
                                                    step_size_up=opts.nbatches_train // 2, step_size_down=None,
                                                    mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle',
                                                    cycle_momentum=False, last_epoch=-1)

    if not opts.cycle_lr:
        lmbda = lambda epoch: opts.learning_rates_mult[epoch]
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)
    opts.optimizer = optimizer
    opts.scheduler = scheduler
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
    loss = opts.loss_fun(opts, inputs, outs)  # Compute the loss.
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


def save_model_and_md(logger: logging, model_fname: str, metadata: dict, epoch: int, opts: argparse) -> None:
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
    logger.info('Saving model to %s' % model_fname)  # Send a message into the logger.
    torch.save({'epoch': epoch, 'model_state_dict': opts.model.state_dict(),
                'optimizer_state_dict': opts.optimizer.state_dict(),
                'scheduler_state_dict': opts.scheduler.state_dict(), 'metadata': metadata, },
               tmp_model_fname)  # Saving the metadata.
    os.rename(tmp_model_fname, model_fname)  # Rename to the desired name.
    logger.info('Saved model')


def load_model(opts: argparse, model_path: str, model_latest_fname: str, gpu=None,
               load_optimizer_and_schedular: bool = False) -> dict:
    """
    Loads and returns the model as a dictionary.
    :param opts: The model opts.
    :param model_path: The model path
    :param model_latest_fname: The model id in the folder.
    :param gpu:
    :param load_optimizer_and_schedular: Whether to load optimizer and schedular.
    :return:
    """
    if gpu is None:
        model_path = os.path.join(model_path, model_latest_fname)
        checkpoint = torch.load(model_path)  # Loading the weights and the metadata.
    else:
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(gpu)
        checkpoint = torch.load(model_path, map_location=loc)
    opts.model.load_state_dict(checkpoint['model_state_dict'])  # Loading the epoch_id, the optimum and the data.
    if load_optimizer_and_schedular:
        opts.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Load the optimizer state.
        opts.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # Load the schedular state.
    return checkpoint


class save_details_class:
    def __init__(self, opts):
        """
        :param opts: The options de decide the metric and the dataset to update the model according to.
        """
        self.optimum = -np.inf  # The initial optimum.
        self.saving_metric = opts.saving_metric  # The metric we should save according to.
        self.dataset_saving_by = opts.dataset_saving_by
        if self.dataset_saving_by == 'train':
            self.dataset_id = 0  # The dataset to save according to.

        elif self.dataset_saving_by == 'test' and (opts.ds_type is DsType.Emnist or opts.ds_type is DsType.FashionMnist) and opts.generelize:
            self.dataset_id = 2
        else:
            self.dataset_id = 1

        if self.saving_metric == 'loss':
            self.metric_idx = 0

        elif self.saving_metric == 'accuracy' and (opts.ds_type is DsType.Emnist or opts.ds_type is DsType.FashionMnist):
            self.metric_idx = 2
        else:
           self.metric_idx = 1

def fit(opts: argparse, the_datasets: list, lan_id:int,direction_id:int) -> None:
    """
    Fitting the model.
    iterate over the datasets and train (or test) them
    :param opts: The model options.
    :param the_datasets: The train, test datasets.
    :param task: The task we learn.
    """
    logger = opts.logger
    #  if opts.first_node:
    logger.info('train_opts: %s', str(opts))
    optimizer = opts.optimizer  # Getting the optimizer.
    scheduler = opts.scheduler  # Getting the scheduler,
    nb_epochs = opts.EPOCHS  # Getting the number of epoch we desire to train for,
    model_dir = opts.model_dir  # Getting the model directory.
    direction = '_right' if direction_id == 0 else '_left'
    model_ext = '.pt'
    model_basename = 'model'
    model_latest_fname = model_basename + '_latest' + direction + model_ext
    model_latest_fname = os.path.join(model_dir, model_latest_fname)
    save_details = save_details_class(opts)  # Contains the optimum.
    last_epoch = -1
    if opts.load_model_if_exists:  # If we want to continue started training.
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
    if instruct(opts, 'abort_after_epochs') and opts.abort_after_epochs > 0:
        end_epoch = st_epoch + opts.abort_after_epochs

    for epoch in range(st_epoch, end_epoch):  # Training for end_Epoch - st_epoch epochs.
        #   if opts.first_node:
        logger.info('Epoch {} learning rate: {}'.format(epoch + 1, optimizer.param_groups[0]['lr']))  # The current lr.
        for the_dataset in the_datasets:  # Training/testing the model by the datasets.
            the_dataset.do_epoch(opts, epoch)
        opts.logger.info('Epoch {} done'.format(epoch + 1))  # logger info done the epoch.
        #
        store_running_stats(opts.model, lan_id,direction_id)  # Storing the running stats to avoid forgetting.
        logger.info('epoch {} done storing running stats direction = {}, language_id = {}'.format(epoch + 1, direction_id, lan_id))  # Adding to the logger.
        #   print("Done storing running stats!")
        #
        for the_dataset in the_datasets:  # Printing the loss, accuracy per dataset.
            logger.info('Epoch {}, {} {}'.format(epoch + 1, the_dataset.name, the_dataset.measurements.print_epoch()))
        if opts.scheduler is not None:  # Scheduler step.
            if not type(opts.scheduler) in [optim.lr_scheduler.CyclicLR, optim.lr_scheduler.OneCycleLR]:
                if epoch < nb_epochs - 1:
                    # learning rate scheduler
                    if type(opts.scheduler) is optim.lr_scheduler.ReduceLROnPlateau:  # Scheduler step in case we reduce lr by loss.
                        the_val_dataset = the_datasets[-1]
                        val_loss = the_val_dataset.measurements.results[-1, 0]
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()

        if opts.save_model:  # Saving the model.
            optimum = save_details.optimum  # Getting the old optimum.
            save_by_dataset = the_datasets[save_details.dataset_id]  # Getting the dataset we update the model according to.
            measurements = np.array(save_by_dataset.measurements.results)
            new_optimum = False  # Flag telling whether we overcome the old optimum.
            epoch_save_value = measurements[epoch, save_details.metric_idx]  # Getting the possible new optimum.
            if epoch_save_value > optimum:  # If we overcome the old optimum.
                optimum = epoch_save_value  # Update the local optimum.
                new_optimum = True  # Change the flag
                logger.info('New optimum: %f' % optimum)  # Adding to the logger.
                save_details.optimum = optimum  # Update the optimum.
            metadata = dict()  # metadata dictionary.
            metadata['epoch'] = epoch
            for the_dataset in the_datasets:  # Storing the loss, accuracy.
                measurements = np.array(the_dataset.measurements.results)
                metadata[the_dataset.name] = measurements
            metadata['optimum'] = optimum  # Storing the new optimum.
            model_latest_fname = model_basename + '_latest' + direction + model_ext
            model_latest_fname = os.path.join(model_dir, model_latest_fname)
            save_model_and_md(logger, model_latest_fname, metadata, epoch,
                              opts)  # Storing the metadata in the model_latest.
            if new_optimum:  # If we have a new optimum, we store in an additional model to avoid overriding.
                model_fname = model_basename + '%d' % (epoch + 1)  + direction + model_ext
                model_fname = os.path.join(model_dir, model_fname)
                shutil.copyfile(model_latest_fname, model_fname)
                logger.info('Saved model to %s' % model_fname)
                #
                model_fname = model_basename + '_best'   + direction + model_ext
                model_fname = os.path.join(model_dir, model_fname)
                shutil.copyfile(model_latest_fname, model_fname)
                logger.info('Saved model to %s' % model_fname)


    logger.info('Done fit')
