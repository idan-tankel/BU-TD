import time

import torch

from v26.funcs import train_step, test_step, logger
from vae.StoreForVae import StoreForVae


class DatasetInfo():
    '''encapsulates a (train/test/validation) dataset with its appropriate train or test function and Measurement class'''

    def __init__(self, istrain, ds, nbatches, name, checkpoints_per_epoch=1):
        self.dataset = ds
        self.nbatches = nbatches
        self.istrain = istrain
        self.checkpoints_per_epoch = checkpoints_per_epoch
        if self.istrain and checkpoints_per_epoch > 1:
            # when checkpoints_per_epoch>1 we make each epoch smaller
            self.nbatches = self.nbatches // checkpoints_per_epoch
        if istrain:
            self.batch_fun = train_step  # This is problematic - without variables...
        else:
            self.batch_fun = test_step  # This is problematic - without variables...
        self.name = name
        self.dataset_iter = None
        self.needinit = True

    def create_measurement(self, measurements_class, model_opts, model):
        """
        create_measurement _summary_

        Args:
            measurements_class (_type_): _description_
            model_opts (_type_): _description_
            model (_type_): _description_
        """        
        self.measurements = measurements_class(model_opts, model)

    def reset_iter(self):
        self.dataset_iter = iter(self.dataset)

    def do_epoch(self, epoch:int, opts, number_of_epochs):
        """
        do_epoch Do a specific epoch of the dataset (validation/training/test)

        Args:
            epoch (int): the current epoch
            opts (SimpleNamespace): The model options
            number_of_epochs (int): The total number of epochs
        """        
        logger.info(self.name)
        nbatches_report = 10
        aborted = False
        self.measurements.reset()
        cur_batches = 0
        if self.needinit or self.checkpoints_per_epoch == 1:
            self.reset_iter()
            self.needinit = False
            if self.istrain and opts.distributed:
                opts.train_sampler.set_epoch(epoch)
                # TODO: when aborted save cur_batches. next, here do for loop and pass over cur_batches
                # and use train_sampler.set_epoch(epoch // checkpoints_per_epoch)
        start_time = time.time()
        for inputs in self.dataset_iter:
            cur_loss, outs = self.batch_fun(inputs, opts)
            with torch.no_grad():
                # so that accuracies calculation will not accumulate gradients
                self.measurements.update(inputs, outs, cur_loss.item())
            cur_batches += 1
            template = 'Epoch {}/ {} step {}/{} {} ({:.1f} estimated minutes/epoch)'
            if cur_batches % nbatches_report == 0:
                duration = time.time() - start_time
                start_time = time.time()
                # print(duration,self.nbatches)
                estimated_epoch_minutes = duration / 60 * self.nbatches / nbatches_report
                logger.info(
                    template.format(epoch + 1, number_of_epochs, cur_batches, self.nbatches,
                                    self.measurements.print_batch(),
                                    estimated_epoch_minutes))

            if True:
                if self.istrain and cur_batches > self.nbatches:
                    aborted = True
                    break

        if not aborted:
            self.needinit = True
        self.measurements.add_history(epoch)
