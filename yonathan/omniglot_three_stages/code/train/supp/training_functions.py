import torch.optim as optim
import logging
import os
import numpy as np
import time
import torch
import shutil
from supp.create_model import *
from supp.batch_norm import *
from torch.utils.data import DataLoader
from supp.loss_and_accuracy import *

#TODO - CHANGE LOAD_MODEL_IF_EXISTS.
class DatasetInfo:
    """encapsulates a (train/test/validation) dataset with its appropriate train or test function and Measurement class"""

    def __init__(self, istrain: bool, data_set: DataLoader, nbatches: int, name: str, checkpoints_per_epoch: int = 1,  sampler=None):
        """
        Args:
            istrain: Whether we should fit the dataset.
            data_set: The data set.
            nbatches: Number of batches in the data_set.
            name: The dataset name.
            checkpoints_per_epoch: Number of checkpoints in the epoch.
            sampler:
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

    def create_measurement(self, measurements_class: type, parser: argparse) -> None:
        """
        We create measurment object to handle our matrcies.
        Args:
            measurements_class: The measurement class should handle the desired matrices.
            parser: The option parser.

        """
        self.measurements = measurements_class(parser)

    def reset_iter(self) -> None:
        """
        crate a dataset iterator.
        """

        self.dataset_iter = iter(self.dataset)

    def do_epoch(self, opts: argparse, epoch: int)->None:
        """
        Trains and tests a model for one epoch.
        Args:
            opts: The model options.
            epoch: The epoch_id.
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
        is_train = self.istrain
        for inputs in self.dataset_iter:  # Iterating over all dataset.
            cur_loss, outs = self.batch_fun(opts, inputs)  # compute the model outputs and the current loss.
            with torch.no_grad():
                # so that accuracies calculation will not accumulate gradients
                self.measurements.update(inputs, outs, cur_loss.item(),is_train)  # update the loss and the accuracy.
            cur_batches += 1  # Update the number of batches.
            template = 'Epoch {} step {}/{} {} ({:.1f} estimated minutes/epoch)'  # Define a convenient template.
            if cur_batches % nbatches_report == 0:
                duration = time.time() - start_time  # Compute the step time.
                start_time = time.time()
                estimated_epoch_minutes = duration / 60 * self.nbatches / nbatches_report  # compute the proportion time.
                opts.logger.info(template.format(epoch + 1, cur_batches, self.nbatches, self.measurements.print_batch(),
                                                 estimated_epoch_minutes))  # Add the epoch_id, loss, accuracy, time for epoch.
        if not aborted:
            self.needinit = True
        self.measurements.add_history(epoch)  # Update the loss and accuracy history.


def train_model(args: argparse, the_datasets: list, learned_params: list, task_id: int) -> None:
    """
    Creating the optimizer & fitting.
    Args:
        args: The model args.
        the_datasets: The datasets.
        learned_params: The learned parameters.
        task_id: The task_id.

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
    fit(args, the_datasets, task_id)


def create_optimizer_and_sched(opts: argparse) -> tuple:
    """
    Args:
        opts: The model options.

    Returns:optimizer, scheduler according to the options.
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
    return optimizer, scheduler


def train_step(opts: argparse, inputs: list[torch]) -> tuple:
    """
    Training the model for one step.
    Args:
        opts: The model options.
        inputs: The model inputs.

    Returns: The loss and output of the batch.
    """
    opts.model.train()  # Move the model into the train mode.
    freeze = opts.freeze
    outs = opts.model(inputs, True, freeze)  # Compute the model output.
    test_all_stages = False
    loss = opts.loss_fun(inputs, outs,test_all_stages)  # Compute the loss.
    opts.optimizer.zero_grad()  # Reset the optimizer.
    loss.backward()  # Do a backward pass.
    opts.optimizer.step()  # Update the model.
    if type(opts.scheduler) in [optim.lr_scheduler.CyclicLR,
                                optim.lr_scheduler.OneCycleLR]:  # Make a scedular step if needed.
        opts.scheduler.step()
    return loss, outs  # Return the loss and the output.


def test_step(opts: argparse, inputs: list[torch]) -> tuple:
    """
    Testing the model for one step.
    Args:
        opts: The model options.
        inputs: The inputs.

    Returns: The loss and the model output.

    """
    opts.model.eval()  # Move the model to evaluation mode in order to not change the running statistics of the batch layers.
    with torch.no_grad():  # Don't need to compute grads.
        freeze = opts.freeze
        outs = opts.model(inputs, False,freeze)  # Compute the model outputs.
        test_all_stages = opts.stages == [0,1,2] # TODO - CHANGE TO LEN.
        loss = opts.loss_fun(inputs, outs,test_all_stages)  # Compute the loss.
    return loss, outs  # Return the loss and the output.

def save_model_and_md(logger: logging, model_fname: str, metadata: dict, epoch: int, opts: argparse) -> None:
    """
    Saving the model weights and the metadata(e.g. running statistics of the batch norm layers)
    Args:
        logger: The logger we write into.
        model_fname: The path to save in.
        metadata: he metadata we desire to save.
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

    # TODO - CHANGE TO BE ADAPTIVE ACCORDING TO THE LOAD_MODEL_IF_EXISTS FLAG.
def load_model(opts: argparse, model_path: str, gpu=None,  load_optimizer_and_schedular: bool = False) -> dict:
    """
    Loading the model state and metadata.
    Args:
        opts: The model options.
        model_path: The model path.
        gpu: Whether we use parallel gpu.
        load_optimizer_and_scheduler: Whether to load optimizer and scheduler states.

    Returns: The loaded data.

    """
    if gpu is None:
     #   model_path = os.path.join(model_path, model_latest_fname)
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

def Load_model_if_exists(opts, logger, model_latest_fname, the_datasets):
    if os.path.exists(model_latest_fname):
        logger.info('Loading model: %s' % model_latest_fname)
        checkpoint = load_model(opts, model_latest_fname, opts.gpu)  # TODO CHANGE TO ONLY OPTS.
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

class save_details_class:
    def __init__(self, opts:argparse):
        """
        Args:
            opts: The model options.
        """
        self.optimum = -np.inf  # The initial optimum.
        self.epoch_save_idx = opts.epoch_save_idx  # The metric we should save according to.
        if opts.dataset_id == 'train':
            self.dataset_id = 0  # The dataset to save according to.
        if opts.dataset_id == 'test':
            self.dataset_id = 1
        else:
            self.dataset_id = 2
        if self.epoch_save_idx == 'loss':
            self.epoch_save_idx = 0
        else:
            self.epoch_save_idx = 1

def fit(opts: argparse, the_datasets: list, task: int) -> None:
    """
    Fitting the model.
      iterate over the datasets and train\test them.
    Args:
        opts: The model options.
        the_datasets: The train, test datasets.
        task: The task we learn.

    """
    logger = opts.logger
    #  if opts.first_node:
    logger.info('train_opts: %s', str(opts))
    optimizer = opts.optimizer  # Getting the optimizer.
    scheduler = opts.scheduler  # Getting the scheduler,
    nb_epochs = opts.EPOCHS  # Getting the number of epoch we desire to train for,
    model_dir = opts.model_dir  # Getting the model directory.
    model_ext = '.pt'
    model_basename = 'model'
    model_latest_fname = model_basename + '_latest' + model_ext
    model_latest_fname = os.path.join(model_dir, model_latest_fname)
    save_details = save_details_class(opts)  # Contains the optimum.
    last_epoch = -1
  #  model_latest_path = model_basename + '_latest' + model_ext
#    if opts.load_model_if_exists:  # If we want to continue started training.
   #     Load_model_if_exists(opts, logger, model_latest_fname, the_datasets)
    st_epoch = last_epoch + 1
    end_epoch = nb_epochs
    if instruct(opts, 'abort_after_epochs') and opts.abort_after_epochs > 0:
        end_epoch = st_epoch + opts.abort_after_epochs

    for epoch in range(st_epoch, end_epoch):  # Training for end_Epoch - st_epoch epochs.
        #   if opts.first_node:
        logger.info('Epoch {} learning rate: {}'.format(epoch + 1, optimizer.param_groups[0]['lr']))  # The current lr.
        for the_dataset in the_datasets:  # Training/testing the model by the datasets.
            the_dataset.do_epoch(opts, epoch)
            if the_dataset.istrain:
             print(accuracy(opts, opts.test_dl, opts.stages))
             store_running_stats(opts.model, task)  # Storing the running stats to avoid forgetting.
             logger.info('epoch {} done storing running stats, task = {}'.format(epoch + 1, task))  # Adding to the logger.

        opts.logger.info('Epoch {} done'.format(epoch + 1))  # logger info done the epoch.
        #
        #  logger.info('epoch {} done storing running stats, task = {}'.format(epoch + 1,task))  # Adding to the logger.
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
            save_by_dataset = the_datasets[
                save_details.dataset_id]  # Getting the dataset we update the model according to.
            measurements = np.array(save_by_dataset.measurements.results)
            new_optimum = False  # Flag telling whether we overcome the old optimum.
            epoch_save_value = measurements[epoch, 3]  # Getting the possible new optimum.
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
            model_latest_fname = model_basename + '_latest' + model_ext
            model_latest_fname = os.path.join(model_dir, model_latest_fname)
            save_model_and_md(logger, model_latest_fname, metadata, epoch,  opts)  # Storing the metadata in the model_latest.
            if new_optimum:  # If we have a new optimum, we store in an additional model to avoid overriding.
                model_fname = model_basename + '%d' % (epoch + 1) + model_ext
                model_fname = os.path.join(model_dir, model_fname)
                shutil.copyfile(model_latest_fname, model_fname)
                logger.info('Saved model to %s' % model_fname)

    logger.info('Done fit')
