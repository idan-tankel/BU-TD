import os.path
import sys
sys.path.append(r'/home/sverkip/data/BU-TD/yonathan/Recognicion/code/')
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Callback
from supp.general_functions import create_optimizer_and_sched
import torch
from supp.Parser import GetParser
from supp.get_dataset import get_dataset_for_spatial_realtions
from supp.FlagAt import Flag, DsType, Model_Options_By_Flag_And_DsType
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers
import torch.optim as optim
from supp.data_functions import preprocess
from pytorch_lightning.loggers import WandbLogger
import numpy as np
import os
import logging


class CheckpointSaver:
    def __init__(self, dirpath, decreasing=True, top_n=5):
        """
        dirpath: Directory path where to store all model weights
        decreasing: If decreasing is `True`, then lower metric is better
        top_n: Total number of models to track based on validation metric value
        """
        if not os.path.exists(dirpath): os.makedirs(dirpath)
        self.dirpath = dirpath
        self.top_n = top_n
        self.decreasing = decreasing
        self.top_model_paths = []
        self.best_metric_val = np.Inf if decreasing else -np.Inf

    def __call__(self, model, epoch, metric_val):
        model_path = os.path.join(self.dirpath, model.__class__.__name__ + f'_epoch{epoch}.pt')
        save = metric_val<self.best_metric_val if self.decreasing else metric_val>self.best_metric_val
        if save:
            logging.info(f"Current metric value better than {metric_val} better than best {self.best_metric_val}, saving model at {model_path}")
            self.best_metric_val = metric_val
            torch.save(model.state_dict(), model_path)
            self.top_model_paths.append({'path': model_path, 'score': metric_val})
            self.top_model_paths = sorted(self.top_model_paths, key=lambda o: o['score'], reverse=not self.decreasing)
        if len(self.top_model_paths) > self.top_n:
            self.cleanup()

    def cleanup(self):
        to_remove = self.top_model_paths[self.top_n:]
        logging.info(f"Removing extra models.. {to_remove}")
        for o in to_remove:
            os.remove(o['path'])
        self.top_model_paths = self.top_model_paths[:self.top_n]

class MyModel(LightningModule):
    def __init__(self, opts, learned_params,ckpt):
        super().__init__()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        self.model = opts.model
        self.opts = opts
        self.loss_fun = opts.loss_fun
        self.ckpt = ckpt
        self.learned_params = learned_params
        self.accuracy = opts.task_accuracy
        self.optimizer , self.scheduler =  create_optimizer_and_sched(self.opts, self.learned_params)
        self.dl = test_dl

    def training_step(self, batch, batch_idx):
        model = self.model
        model.train()  # Move the model into the train mode.
        outs = model(batch)  # Compute the model output.
        loss = self.loss_fun( batch, outs)  # Compute the loss.
        self.optimizer.zero_grad()  # Reset the optimizer.
        loss.backward()  # Do a backward pass.
        self.optimizer.step()  # Update the model.
        samples = self.opts.inputs_to_struct(batch)
        outs = self.model.outs_to_struct(outs)

        _ , acc = self.accuracy(outs, samples)
        if type(self.scheduler) in [optim.lr_scheduler.CyclicLR, optim.lr_scheduler.OneCycleLR]:  # Make a scheduler step if needed.
            self.scheduler.step()
       # print(self.log)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger = True)
        self.log('train_acc',acc, on_step=True, on_epoch=True, logger=True)

        return loss  # Return the loss and the output.

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            outs = self.model(batch)
            loss = self.loss_fun(batch, outs)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            outs = self.model(batch)
            loss = self.loss_fun(batch, outs)  # Compute the loss.
            outs = self.model.outs_to_struct(outs)
            samples =opts.inputs_to_struct(batch)
            _ , task_accuracy = self.accuracy(outs, samples)
            batch_size = batch[0].shape[0]
            samples = self.opts.inputs_to_struct(batch)
            _ , acc = self.accuracy(outs, samples)
            self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True)
            self.log('val_acc', task_accuracy, on_step=True, on_epoch=True, logger=True)
         #   print(task_accuracy)
            return task_accuracy.sum()

    def configure_optimizers(self):
        opti, sched = create_optimizer_and_sched(self.opts, self.learned_params)
        # return {"optimizer":opti, "lr_scheduler":sched}
        return [opti], [sched]

    def validation_epoch_end(self, outputs):
        acc = sum(outputs) / len(outputs)
        print(acc)
        self.ckpt(self.model,self.current_epoch,acc)
        return sum(outputs) / len(outputs)

    def test_epoch_end(self, outputs):
        print(self.accuracy_dl(self.dl))
        return torch.cat(outputs, dim=0).sum() / (len(outputs))

    def accuracy_dl(self,dl):
        acc = 0.0
        for inputs in dl:
            inputs = preprocess(inputs)
            outs = self.model(inputs)
            samples = self.opts.inputs_to_struct(inputs)
            outs = self.model.outs_to_struct(outs)
            pred , acc_batch = self.accuracy(outs, samples)
            acc += acc_batch
        acc = acc / len(dl)
        return acc

opts = Model_Options_By_Flag_And_DsType(Flag=Flag.NOFLAG, DsType=DsType.Emnist)
parser = GetParser(opts=opts, language_idx=0,direction = 'right')
data_path = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/samples/18_extended'
# Create the data for right.
tmpdir = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/results/'

[the_datasets, train_dl ,  test_dl, val_dl , _ , _ , _ ] = get_dataset_for_spatial_realtions(parser, data_path,lang_idx = 0, direction = 2)
#model_ckpt = ModelCheckpoint(dirpath=tmpdir, monitor=None, save_top_k=-1, save_last=True)
learned_params = list((parser.model.parameters()))
ModelCkpt = ModelCheckpoint(dirpath='/home/sverkip/data/BU-TD/Recognicion/data/emnist/results/MyFirstCkpt',monitor='val_acc', filename='First_model')
Checkpoint_saver = CheckpointSaver(dirpath = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/results/MyFirstCkpt', decreasing=False, top_n=5)
wandb_logger = WandbLogger(project = "My_first_project_5.10", job_type = 'train',save_dir='/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/results/')
trainer = pl.Trainer(accelerator='gpu',max_epochs = 60,logger=wandb_logger,callbacks=[ModelCkpt])
model = MyModel(parser,learned_params,ckpt = Checkpoint_saver)
#path ="/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/results/last-v12.ckpt"
#model = model.load_from_checkpoint(path,opts = parser,learned_params = learned_params)
#print(model.accuracy_dl(test_dl))
trainer.fit(model, train_dataloaders = train_dl, val_dataloaders = test_dl)

# Goals:
# 1. having train -> test - > val.
# 2. have full logger.
# 3. save model in checkpoints by test.
# 4. Get rid of training_functions, measurmnets.