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
        model.to(device("cuda") if cuda.is_available() else device("cpu"))
        if hasattr(config, 'Models'):
            config = config.Models
        model.classifier = nn.Sequential(
            nn.Linear(768, config.num_classes * config.number_of_linear_heads),
            nn.Unflatten(
                1, (config.num_classes, config.number_of_linear_heads)),
        )
        self.model = model

    def training_step(self, batch, batch_index):
        x, y = batch['img'], batch['label_all']
        x_hat = self.model(x)
        y.squeeze_()
        y.to(device("cuda") if cuda.is_available() else device("cpu"))
        loss = nn.CrossEntropyLoss(reduction='mean')(x_hat.logits, y)
        acc = 1 - (x_hat.logits.argmax(dim=1) - y).count_nonzero() / y.numel()
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
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
            x, y = batch['img'], batch['label_all']
            x_hat = self.model(x)
            y.squeeze_()
            y.to(device("cuda") if cuda.is_available() else device("cpu"))
            loss = nn.CrossEntropyLoss(reduction='mean')(x_hat.logits, y)
            acc = 1 - (x_hat.logits.argmax(dim=1) -
                       y).count_nonzero() / y.numel()
            self.log('val_loss', loss, on_step=True,
                     on_epoch=True, logger=True)
            self.log('val_acc', acc, on_step=True, on_epoch=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer
