import os.path

import pytorch_lightning as pl
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

class ImagePredictionLogger(Callback):
    def __init__(self, val_samples, num_samples = 2000):
        super().__init__()
        self.num_samples = num_samples
        self.val_samples = val_samples



class MyModel(LightningModule):
    def __init__(self, opts, learned_params):
        super().__init__()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        self.model = opts.model
        self.opts = opts
        self.loss_fun = opts.loss_fun
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
            return task_accuracy.sum()/batch_size

    def configure_optimizers(self):
        opti, sched = create_optimizer_and_sched(self.opts, self.learned_params)
        # return {"optimizer":opti, "lr_scheduler":sched}
        return [opti], [sched]

    def validation_epoch_end(self, outputs):
        print(self.accuracy_dl(self.dl))
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

opts = Model_Options_By_Flag_And_DsType(Flag=Flag.ZF, DsType=DsType.Emnist)
parser = GetParser(opts=opts, language_idx=0,direction = 'right')
data_path = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/samples/18_extended'
# Create the data for right.
tmpdir = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/results/'

[the_datasets, train_dl ,  test_dl, val_dl , _ , _ , _ ] = get_dataset_for_spatial_realtions(parser, data_path,lang_idx = 0, direction = 0)
#model_ckpt = ModelCheckpoint(dirpath=tmpdir, monitor=None, save_top_k=-1, save_last=True)
learned_params = list((parser.model.parameters()))
wandb_logger = WandbLogger(project = "My_first_project_5.10", job_type = 'train',save_dir='/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/results/MyFirstCheckpoint.ckpt')
trainer = pl.Trainer(accelerator='gpu',max_epochs = 60,logger=wandb_logger,default_root_dir = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/results/Myckpt.ckpt/')
model = MyModel(parser,learned_params)
#path ="/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/results/last-v12.ckpt"
#model = model.load_from_checkpoint(path,opts = parser,learned_params = learned_params)
#print(model.accuracy_dl(test_dl))
trainer.fit(model, train_dataloaders = train_dl, val_dataloaders = test_dl)

# Goals:
# 1. having train -> test - > val.
# 2. have full logger.
# 3. save model in checkpoints by test.
# 4. Get rid of training_functions, measurmnets.