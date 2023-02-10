# this file represent a switcher for the models and as interface
import pytorch_lightning as pl
from torch import nn, device, cuda
import torch
from transformers import ViTForImageClassification, ViTConfig
import torch.nn.functional as F
from torch.optim import Adam
from torch import no_grad


class ModelWrapper(
        pl.LightningModule):
    """
    ModelWrapper Is a wrapper for the model that we want to train via pytorch lightning
    2 important things here:
    1. The model is wrapped with the pytorch lightning module, which will be modified if multilabel classification is needed according to number of heads
    2. The implementation of the training step and validation step, including using the custom data loader we have made, and the loss functions

    Returns:
        `pl.LightningModule`: The modified model
    """

    def __init__(self, model, config):
        super().__init__()
        # self.loss = nn.CrossEntropyLoss()
        self.task = config.Training.task
        model.to(device("cuda") if cuda.is_available() else device("cpu"))
        if hasattr(config, 'Models'):
            config = config.Models
        model.classifier = nn.Sequential(
            nn.Linear(768, config.num_classes * config.number_of_linear_heads),
            nn.Unflatten(
                1, (config.num_classes, config.number_of_linear_heads)),
        )
        # this is basically a respahe but wanted the operation as a layer
        self.model = model

    def training_step(self, batch, batch_index):
        # self.model.train()
        if self.task == "multi_label_classification":
            x, y = batch['img'], batch['label_all']
            batch_size = y.shape[0]
            y = y.reshape(batch_size,1) 
            # for CE loss, we will fold up the other dimention if the examples are grid
        elif self.task == "vanilla_training":
            x,y = batch
            
        else:
            x, y = batch['img'], batch['label_task'].gather(index=batch['label_all'].squeeze(1),dim=1)
            y.squeeze_()
        x_hat = self.model(x)
        y.to(device("cuda") if cuda.is_available() else device("cpu"))
        x_hat.logits.squeeze_()
        y.squeeze_()
        loss = nn.CrossEntropyLoss(reduction='mean')(x_hat.logits, y)
        acc = 1 - (x_hat.logits.argmax(dim=1) - y).count_nonzero() / y.numel()
        pred = x_hat.logits.argmax(dim=1)
        non_naive_ratio = (pred != 47).sum() / pred.numel()
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
        self.log(name='non_naive_ratio',value=non_naive_ratio, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_index):
        """
        validation_step This is implementation of the dataset 

        Args:
            batch (Torch.Tensor): The batch of the data
            batch_index (`int`):  The index of the batch

        Returns:
            float: The loss of the batch
        """
        self.model.eval()
        with no_grad():
            if self.task == "multi_label_classification":
                x, y = batch['img'], batch['label_all']
                batch_size = y.shape[0]
                y = y.reshape(batch_size,1) 
                # for CE loss, we will fold up the other dimention if the examples are grid
            elif self.task == "vanilla_training":
                x,y = batch
            else:
                x, y = batch['img'], batch['label_task'].gather(index=batch['label_all'].squeeze(1),dim=1)
                y.squeeze_()
            x_hat = self.model(x)
            y.to(device("cuda") if cuda.is_available() else device("cpu"))

            loss = nn.CrossEntropyLoss(reduction='mean')(x_hat.logits.squeeze(), y.squeeze())
            pred = x_hat.logits.argmax(dim=1)
            acc = 1 - (pred -
                       y).count_nonzero() / y.numel()
            # this is the distance from the naive classifier - classifing all to the most common class - edge case
            # 47 is the enumeration of the N.A class
            non_naive_ratio = (pred != 47).sum() / pred.numel()
            self.log('val_loss', loss, on_step=True,
                     on_epoch=True, logger=True)
            self.log('val_acc', acc, on_step=True, on_epoch=True, logger=True)
            self.log(name='non_naive_ratio',value=non_naive_ratio, on_step=True, on_epoch=True, logger=True)
        return loss

    def configure_optimizers(self):
        batch_size = 10
        optimizer = Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer=optimizer,
            base_lr = 0.0001,
            max_lr = 0.002,
            step_size_up=batch_size//2,
            step_size_down=None,
            mode='triangular',
            gamma=1.0,
            scale_fn=None,
            scale_mode='cycle',
            cycle_momentum=False,
            last_epoch=-1)
        monitor = 'val_loss'
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": monitor}

