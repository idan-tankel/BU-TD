import logging
import os
import numpy as np
import torch.optim as optim
import argparse
import time
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from supp.FlagAt import  DsType
from supp.batch_norm import store_running_stats
import copy

class DatasetInfo:
    """encapsulates a (train/test/validation) dataset with its appropriate train or test function and Measurement class"""

    def __init__(self, istrain: bool, data_set: DataLoader, nbatches: int, name: str, checkpoints_per_epoch: int = 1, sampler=None):
        """
        Args:
            istrain: Whether we should fit the dataset.
            data_set:  The data set.
            nbatches: Number of batches in the data_set.
            name: The dataset name.
            checkpoints_per_epoch: Number of checkpoints in the epoch..
            sampler: The sampler.
        """
        self.dataset = data_set
        self.nbatches = nbatches
        self.istrain = istrain
        self.checkpoints_per_epoch = checkpoints_per_epoch
        if self.istrain and checkpoints_per_epoch > 1:
            self.nbatches = self.nbatches // checkpoints_per_epoch
        if istrain:  # If we fit the data_loader we choose each step to be train_step including backward step,schedular step.
            self.batch_fun = train_step
        else:  # otherwise we just do a forward pass and don't update the model and the scheduler.
            self.batch_fun = test_step
        self.name = name
        self.dataset_iter = None
        self.needinit = True
        self.sampler = sampler

    def create_measurement(self, measurements_class: type, opts: argparse, model: nn.Module) :
        """
        We create measurment object to handle our matrices.
        Args:
            measurements_class: The measurement class should handle the desired matrices.
            opts: The model options.
            model: The model.

        """
        self.measurements = measurements_class(opts, model)

    def reset_iter(self) :
        """
        crate a dataset iterator.
        """
        self.dataset_iter = iter(self.dataset)

    def do_epoch(self, opts: argparse, epoch: int)->None:
        """
        Args:
            opts: The model options.
            epoch: The epoch id.

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
                opts.logger.info(template.format(epoch + 1, cur_batches, self.nbatches, self.measurements.print_batch(), estimated_epoch_minutes))  # Add the epoch_id, loss, accuracy, time for epoch.

        if not aborted:
            self.needinit = True
        self.measurements.add_history(epoch)  # Update the loss and accuracy history.

def train_step(opts: argparse, inputs: list[torch]) -> tuple:
    """
    Making a train step in which we update the model.
    Args:
        opts: The model options.
        inputs: The model inputs.

    Returns: The loss on the batch, the model outs.

    """
    opts.model.train()  # Move the model into the train mode.
    outs = opts.model(inputs)  # Compute the model output.
    loss = opts.loss_fun(opts.model, inputs, outs)  # Compute the loss.
    opts.optimizer.zero_grad()  # Reset the optimizer.
    loss.backward()  # Do a backward pass.
    opts.optimizer.step()  # Update the model.
    if type(opts.scheduler) in [optim.lr_scheduler.CyclicLR, optim.lr_scheduler.OneCycleLR]:  # Make a scedular step if needed.
        opts.scheduler.step()
    return loss, outs  # Return the loss and the output.

def test_step(opts: argparse, inputs: list[torch]) -> tuple:
    """
    Making a test step in which we don't update the model.
    Args:
        opts: The model options.
        inputs: The model inputs.

    Returns: The loss on the batch, the model outs.

    """
    opts.model.eval()  # Move the model to evaluation mode in order to not change the running statistics of the batch layers.
    with torch.no_grad():  # Don't need to compute grads.
        outs = opts.model(inputs)  # Compute the model outputs.
        loss = opts.loss_fun(opts.model, inputs, outs)  # Compute the loss.
    return loss, outs  # Return the loss and the output.

def save_model_and_md(logger: logging, model_fname: str, metadata: dict, epoch: int, opts: argparse) -> None:
    """
    Saving the model weights and the metadata(e.g. running statistics of the batch norm layers)
    Args:
        logger: The logger we write into.
        model_fname: The path to save in.
        metadata: The metadata we desire to save.
        epoch: The epoch_id.
        opts: The model options.

    """
    tmp_model_fname = model_fname + '.tmp'  # Save into temporal directory,
    logger.info('Saving model to %s' % model_fname)  # Send a message into the logger.
    torch.save({'epoch': epoch, 'model_state_dict': opts.model.state_dict(),
                'optimizer_state_dict': opts.optimizer.state_dict(),
                'scheduler_state_dict': opts.scheduler.state_dict(), 'metadata': metadata, },
               tmp_model_fname)  # Saving the metadata.
    os.rename(tmp_model_fname, model_fname)  # Rename to the desired name.
    logger.info('Saved model')


def Change_checkpoint(checkpoint, model, ntasks, ndirection):
    check_new = {}
    for param in checkpoint.keys():
        if not 'module.Head.taskhead' in param:
            check_new[param] = checkpoint[param]

    for i in range(ntasks*ndirection):
      param ="module.Head.taskhead."+str(i)+".layers.0.weight"
      check_new[param] = model.state_dict()[param]
      param = "module.Head.taskhead." + str(i) + ".layers.0.bias"
      check_new[param] = model.state_dict()[param]
    '''
    for param in checkpoint.keys():
        if 'norm.running_var' or param or 'norm.running_mean' in param:
          check_new[param] = model.state_dict()[param]
    '''
    return check_new


# TODO - MOVE IT TO GENERAL FUNCTIONS.
def load_model(model: nn.Module, model_path: str, model_latest_fname: str, gpu=None, load_optimizer_and_schedular: bool = False) -> dict:
    """
    Loads and returns the model as a dictionary.
    Args:
        opts: The model options.
        model_path: The path to the model.
        model_latest_fname:  The model id in the folder.
        gpu:
        load_optimizer_and_scheduler:  Whether to load optimizer and scheduler.

    Returns: The loaded checkpoint.

    """
    if gpu is None:
        model_path = os.path.join(model_path, model_latest_fname)
        checkpoint = torch.load(model_path)  # Loading the weights and the metadata.
    else:
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(gpu)
        checkpoint = torch.load(model_path, map_location=loc)
    new_load = False
    if new_load:
     checkpoint = Change_checkpoint( checkpoint['model_state_dict'],model, 51,4)
    else:
     checkpoint = checkpoint['model_state_dict']
    model.load_state_dict(checkpoint)  # Loading the epoch_id, the optimum and the data.
    if load_optimizer_and_schedular:
        opts.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Load the optimizer state.
        opts.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # Load the schedular state.
    return checkpoint

class save_details_class:
    def __init__(self, opts:argparse):
        """
        Args:
            opts: The model options.
        """
        self.optimum = -np.inf  # The initial optimum.
        self.saving_metric = opts.saving_metric  # The metric we should save according to.
        self.dataset_saving_by = opts.dataset_saving_by
        if self.dataset_saving_by == 'train':
            self.dataset_id = 0  # The dataset to save according to.

        elif self.dataset_saving_by == 'test' and (opts.ds_type is DsType.Emnist or opts.ds_type is DsType.FashionMnist):
            self.dataset_id = 1
        else:
            self.dataset_id = 1

        if self.saving_metric == 'loss':
            self.metric_idx = 0

        elif self.saving_metric == 'accuracy' and (opts.ds_type is DsType.Emnist or opts.ds_type is DsType.FashionMnist):
            self.metric_idx = 2
        else:
           self.metric_idx = 1

    def update(self,new_optimum):
        self.optimum = new_optimum

def fit(opts: argparse, the_datasets: list, task: int,direction_id:int) -> save_details_class:
    """
    Args:
        opts: The model options.
        the_datasets: The dataset.
        task: The task id.
        direction_id: The direction id.

    Returns: The save model details.

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

    for epoch in range(st_epoch, end_epoch):  # Training for end_Epoch - st_epoch epochs.
        #   if opts.first_node:
        logger.info('Epoch {} learning rate: {}'.format(epoch + 1, optimizer.param_groups[0]['lr']))  # The current lr.
        for the_dataset in the_datasets:  # Training/testing the model by the datasets.
            the_dataset.do_epoch(opts, epoch)
        opts.logger.info('Epoch {} done'.format(epoch + 1))  # logger info done the epoch.
        #
      #  store_running_stats(opts.model, task)  # Storing the running stats to avoid forgetting.
      #  logger.info('epoch {} done storing running stats task = {}, direction = {}'.format(epoch + 1,task,direction_id))  # Adding to the logger.
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
                save_details.update(optimum)  # Update the optimum.
            metadata = dict()  # metadata dictionary.
            metadata['epoch'] = epoch
            for the_dataset in the_datasets:  # Storing the loss, accuracy.
                measurements = np.array(the_dataset.measurements.results)
                metadata[the_dataset.name] = measurements
            metadata['optimum'] = optimum  # Storing the new optimum.
            model_latest_fname = model_basename + '_latest' + direction + model_ext
            model_latest_fname = os.path.join(model_dir, model_latest_fname)
            save_model_and_md(logger, model_latest_fname, metadata, epoch, opts)  # Storing the metadata in the model_latest.
            if new_optimum:  # If we have a new optimum, we store in an additional model to avoid overriding.
                model_fname = model_basename + '%d' % (epoch + 1)  + direction + model_ext
                model_fname = os.path.join(model_dir, model_fname)
                shutil.copyfile(model_latest_fname, model_fname)
                logger.info('Saved model to %s' % model_fname)
                 #
                model_fname = model_basename +  direction + '_best' +model_ext
                model_fname = os.path.join(model_dir, model_fname)
                shutil.copyfile(model_latest_fname, model_fname)
                logger.info('Saved model to %s' % model_fname)
                 #

    logger.info('Done fit')
    return save_model_and_md
