import os.path
import sys
sys.path.append(r'/home/sverkip/data/BU-TD/yonathan/Recognicion/code/')
from pytorch_lightning import LightningModule
from supp.general_functions import create_optimizer_and_sched
import torch
import torch.optim as optim
from supp.general_functions import preprocess
import numpy as np
import os
import logging
import torch.nn as nn

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

    def __call__(self, model, epoch, metric_val,optimizer,scheduler, parser):
        model_path = os.path.join(self.dirpath, model.__class__.__name__ + f'_epoch{epoch}.pt')
        save = metric_val<self.best_metric_val if self.decreasing else metric_val>self.best_metric_val
        if save:
            logging.info(f"Current metric value better than {metric_val} better than best {self.best_metric_val}, saving model at {model_path}")
            self.best_metric_val = metric_val
            save_data = {'epoch': epoch, 'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'scheduler_state_dict': scheduler.state_dict(), 'parser': parser }
            torch.save(save_data, model_path)
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

class ModelWrapped(LightningModule):
    def __init__(self, opts, learned_params, ckpt):
        super().__init__()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        self.model = opts.model
        self.opts = opts
        self.loss_fun = opts.criterion
        # change this! the saving of something under the opts object without a reason is harder to understand
        self.ckpt = ckpt
        self.learned_params = learned_params
        self.accuracy = opts.task_accuracy
        self.optimizer , self.scheduler =  create_optimizer_and_sched(self.opts, self.learned_params)

    def training_step(self, batch, batch_idx):
        model = self.model
        model.train()  # Move the model into the train mode.
        outs = model(batch)  # Compute the model output.
        loss = self.loss_fun( opts=self.opts,inputs=batch, outs=outs)  # Compute the loss.
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
            loss = self.loss_fun(opts=self.opts,inputs=batch, outs=outs)  # Compute the loss.
            outs = self.model.outs_to_struct(outs)
            samples = self.opts.inputs_to_struct(batch)
            _ , task_accuracy = self.accuracy(outs, samples)
            samples = self.opts.inputs_to_struct(batch)
            _ , acc = self.accuracy(outs, samples)
            self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True)
            self.log('val_acc', task_accuracy, on_step=True, on_epoch=True, logger=True)
            return task_accuracy.sum()

    def configure_optimizers(self):
        opti, sched = create_optimizer_and_sched(self.opts, self.learned_params)
        return [opti], [sched]

    def validation_epoch_end(self, outputs):
        acc = sum(outputs) / len(outputs)
        print(acc)
        if self.ckpt != None:
          self.ckpt(self.model,self.current_epoch,acc,self.optimizer,self.scheduler, self.opts)
        return sum(outputs) / len(outputs)

    def test_epoch_end(self, outputs):
        print(self.accuracy_dl(self.dl))
        return torch.cat(outputs, dim=0).sum() / (len(outputs))

    def accuracy_dl(self, dl):
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

class Training_flag:
    def __init__(self,parser, train_all_model: bool, train_arg: bool, train_task_embedding: bool, train_head: bool):
        """
        Args:
            train_all_model: Whether to train all model.
            train_arg: Whether to train arg.
            task_embedding: Whether to train the task embedding.
            head_learning: Whether to train the read out head.
        """
        self.train_all_model = train_all_model
        self.train_arg = train_arg
        self.task_embedding = train_task_embedding
        self.head_learning = train_head
        self.parser= parser

    def Get_learned_params(self, model: nn.Module, lang_idx: int, direction: int):
        """
        Args:
            model: The model.
            lang_idx: Language index.
            direction: The direction.

        Returns: The desired parameters.
        """
        if self.train_all_model:
            return list(model.parameters())
        idx = lang_idx * self.parser.ndirections + direction
        learned_params = []
        if self.task_embedding:
            learned_params.extend(model.task_embedding[direction])
        if self.head_learning:
            learned_params.extend(model.transfer_learning[idx])
        if self.train_arg:
            learned_params.extend(model.tdmodel.argument_embedding[lang_idx])
        return learned_params