# this file represent a switcher for the models and as interface
import pytorch_lightning as pl
from torch import nn, device, cuda
from transformers import ViTForImageClassification, ViTConfig
import torch.nn.functional as F
from torch.optim import Adam
from torch import no_grad


class ModelWrapper(
        pl.LightningModule):
    """
    ModelWrapper Is a wrapper for the model that we want to train via pytorch lightning
    Returns:
        _type_: _description_
    """

    def __init__(self, model):
        super().__init__()
        # self.loss = nn.CrossEntropyLoss()
        model.to(device("cuda") if cuda.is_available() else device("cpu"))
        model.classifier = nn.Linear(768, 101, bias=True)
        self.model = model

    def training_step(self, batch,batch_index):
        x, y = batch
        x_hat = self.model(x)
        loss = F.cross_entropy(x_hat.logits, y)
        acc = (x_hat.logits.argmax(dim=1) - y).count_nonzero() / y.numel()
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch,batch_index):
        """
        validation_step This is implementation of the dataset 

        Args:
            batch (_type_): _description_
            batch_index (_type_): _description_

        Returns:
            _type_: _description_
        """        
        with no_grad():
            x, y = batch
            x_hat = self.model(x)
            loss = F.cross_entropy(x_hat.logits, y)
            acc = (x_hat.logits.argmax(dim=1) - y).count_nonzero() / y.numel()
            self.log('val_loss', loss, on_step=True,
                     on_epoch=True, logger=True)
            self.log('val_acc', acc, on_step=True, on_epoch=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer
