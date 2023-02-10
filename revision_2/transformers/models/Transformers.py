import os
from itertools import chain

import pytorch_lightning as pl
import timm
import torch
import wandb
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import Resize

from common.common_classes import DataInputs
from common.common_functions import get_inputs_from_list
from transformers.Utils.TrainingData import TrainData
from transformers.configs.config import Config, Dataset
from transformers.models.timm_models.factory import create_model as create_timm_model
from transformers.objectss_of_tasks.ObjectTaskInterface import ObjectTaskInterface


def apply_pretrained_weights(pretrained_model: torch.nn.Module, made_model_transformer: torch.nn.Module):
    # for (pretrained), (made) in zip(pretrained_model.named_parameters(), made_model_transformer.named_parameters()):
    for (pretrained), (made) in zip(pretrained_model.state_dict().items(), made_model_transformer.state_dict().items()):
        pretrained_name, pretrained_param = pretrained
        made_name, made_param = made

        if made_param.shape == pretrained_param.shape:
            made_param.copy_(pretrained_param)
        else:
            smaller = [(0, m - p) for m, p in zip(made_param.shape, pretrained_param.shape)]
            smaller.reverse()
            pad = tuple(chain.from_iterable(smaller))

            padded_tensor = torch.nn.functional.pad(pretrained_param, pad)
            not_zero = padded_tensor != 0

            made_param_clone = made_param.clone()

            # made_param_clone in not_zero set to zero
            made_param_clone[not_zero] = 0
            made_param_clone = made_param_clone + padded_tensor

            made_param.copy_(made_param_clone)


class MultiHeadsOutputVit(pl.LightningModule):
    def __init__(self, model_implementation: str, model_input_shape, number_of_classes: int,
                 number_of_heads: int, config: Config, mean: Tensor, std: Tensor, object_task: ObjectTaskInterface,
                 train_data: TrainData, batch_size: int, num_instructions: int):
        # Now we need to create model-layer from the image size - to the made_model input size
        super().__init__()
        self.batch_size = batch_size
        self.train_data: TrainData = train_data
        self.save_hyperparameters()
        self.model_input_shape = model_input_shape
        self.object_task: ObjectTaskInterface = object_task

        self.transforms_resize: Resize = object_task.reshape_input(self.model_input_shape[1:3])

        # TODO =continue this
        pretrained_model = timm.create_model(model_implementation, pretrained=True, num_classes=object_task.output_size)
        self.made_model_transformer = create_timm_model(model_implementation, pretrained=False,
                                                        num_classes=object_task.output_size,
                                                        num_instructions=num_instructions)
        apply_pretrained_weights(pretrained_model, self.made_model_transformer)
        # self.made_model_transformer = models.create_model(model_implementation, pretrained=True,
        #                                                   num_classes=object_task.output_size)
        # self.add_tokens_to_model(num_instructions)
        self.another_layers = object_task.another_layers(num_instructions, self.made_model_transformer)

        self.object_task.made_model_transformer = self.made_model_transformer
        self.object_task.transforms_resize = self.transforms_resize

        self.config = config
        self.chosen_dataset: Dataset = config.datasets_specs.chosen_dataset
        self.number_of_classes = number_of_classes
        self.mean = mean
        self.std = std
        self.lr: float = config.running_specification.model.training_options_config.initial_lr
        # TODO add: lr, weight_decay

    def add_tokens_to_model(self, num_instructions: int):
        starting_embedding_size: int = self.get_model_init_size()
        for param in self.made_model_transformer.parameters():

            if isinstance(param, torch.nn.parameter.Parameter):
                pass

    def get_model_init_size(self) -> int:  # TODO - change to get_model_init_size
        starting_embedding_size = self.made_model_transformer.transformers[0].blocks[0].norm1.normalized_shape[0]
        return starting_embedding_size

    def forward(self, img):
        x = self.transforms_resize(img)
        # TODO - concatenate the instructions

        x = self.made_model_transformer(x)  # The Vit model
        return x

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        task_accuracy, loss = self.get_loss_and_acc(batch)
        wandb.log({'train_step_loss': loss, 'train_step_acc': task_accuracy})
        return {'loss': loss, 'acc': task_accuracy}

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-4, amsgrad=True)  # TODO - from config)
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        dl = self.get_dl('train', self.config.running_specification.running.nsamples_train, shuffle=True)
        return dl

    def validation_step(self, batch, batch_idx):
        task_accuracy, loss = self.get_loss_and_acc(batch)
        tensorboard_logs = {'val_loss': loss, 'val_acc': task_accuracy}
        wandb.log({'val_step_loss': loss, 'val_step_acc': task_accuracy})
        return {'val_loss': loss, 'val_acc': task_accuracy}

    def val_dataloader(self):
        dl = self.get_dl('val', self.config.running_specification.running.nsamples_val)

        return dl

    def test_step(self, batch, batch_idx):
        task_accuracy, loss = self.get_loss_and_acc(batch)
        tensorboard_logs = {'test_loss': loss, 'test_acc': task_accuracy}
        return {'test_loss': loss, 'test_acc': task_accuracy}

    def test_dataloader(self):
        dl = self.get_dl('test', self.config.running_specification.running.nsamples_test)

        return dl

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'].mean() for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'].mean() for x in outputs]).mean()
        self.log('train_loss', avg_loss, sync_dist=True)
        self.log('train_acc', avg_acc, sync_dist=True)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'].mean() for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'].mean() for x in outputs]).mean()
        self.log('val_loss', avg_loss, sync_dist=True)
        self.log('val_acc', avg_acc, sync_dist=True)

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'].mean() for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'].mean() for x in outputs]).mean()
        self.log('test_loss', avg_loss, sync_dist=True)
        self.log('test_acc', avg_acc, sync_dist=True)

    def get_dl(self, name: str, nexamples: int, shuffle: bool = False) -> DataLoader:
        dataset_function = self.chosen_dataset.dataset_function
        dataset = dataset_function(
            root=os.path.join(self.chosen_dataset.chosen_dataset_full_path, name),
            nclasses_existence=self.chosen_dataset.nclasses_existence,
            ndirections=self.config.running_specification.running.ndirections,
            nexamples=nexamples,
            split=True,
            mean=self.mean,
            std=self.std)
        dl: DataLoader = DataLoader(dataset,
                                    batch_size=self.config.running_specification.running.batch_size,
                                    num_workers=self.config.running_specification.running.num_workers,
                                    shuffle=shuffle,
                                    pin_memory=True)
        return dl

    def get_loss_and_acc(self, batch):
        inputs: DataInputs = get_inputs_from_list(batch)
        # Forward pass
        output = self.object_task.forward(inputs, self.another_layers)
        # output = self(inputs.images)
        loss = self.object_task.loss(inputs, output, self.number_of_classes, self.train_data.is_pretrain)
        # loss = self.object_task.loss(inputs.label_existence.squeeze(1), output, self.number_of_classes)
        task_accuracy = self.object_task.accuracy(output, inputs, self.number_of_classes, self.train_data.is_pretrain)

        self.train_data.acc = task_accuracy.mean().item()
        self.train_data.loss = loss.mean().item()

        return task_accuracy, loss
