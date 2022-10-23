import sys
sys.path.append(r'/home/idanta/BU-TD/yonathan/Recognicion/code/')
import imp
# TODO write an own import function using imp
from supp.Dataset_and_model_type_specification import Flag
import torch.nn as nn
from Checkpoint_model_definition import CheckpointSaver, ModelWrapped
from pathlib import Path
import os
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from supp.get_dataset import get_dataset_for_spatial_realtions
from supp.Parser import GetParser
import pytorch_lightning as pl


class Training_flag:
    def __init__(self, train_all_model: bool, train_arg: bool, train_task_embedding: bool, train_head: bool):
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
        idx = lang_idx * 4 + direction
        learned_params = []
        if self.task_embedding:
            learned_params.extend(model.task_embedding[direction])
        if self.head_learning:
            learned_params.extend(model.transfer_learning[idx])
        if self.train_arg:
            learned_params.extend(model.tdmodel.argument_embedding[lang_idx])
        return learned_params


def main(train_right, train_left):
    project_path = Path(__file__).parents[2]
    data_path = os.path.join(
        project_path, 'data/emnist/samples/24_extended_testing')
    # TODO change these hard coded paths!
    tmpdir = os.path.join(project_path, 'data/emnist/results/')
    checkpoint_path = os.path.join(tmpdir, 'MyFirstCkt.ckpt')
    parser = GetParser(task_idx=0, direction_idx='right', flag=Flag.ZF)
  #  [the_datasets, train_dl, test_dl, val_dl, _, _, _] = get_dataset_for_spatial_realtions(parser, data_path,          lang_idx=0, direction=0)
    ModelCkpt = ModelCheckpoint(
        dirpath=tmpdir, monitor="val_loss_epoch", mode="min")
    Checkpoint_saver = CheckpointSaver(
        dirpath=checkpoint_path, decreasing=False, top_n=5)
    # TODO: remove this classs since the checkpoint is already defined
    wandb_logger = WandbLogger(project="My_first_project_5.10", job_type='train',
                               save_dir='/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/results/')
    trainer = pl.Trainer(accelerator='gpu', max_epochs=60,
                         logger=wandb_logger, callbacks=[ModelCkpt])
    training_flag = Training_flag(
        train_all_model=True, train_arg=False, train_task_embedding=False, train_head=False)
    learned_params = training_flag.Get_learned_params(
        parser.model, lang_idx=0, direction=0)
    if train_right:
        train_dl, test_dl, val_dl, *the_datasets = get_dataset_for_spatial_realtions(
                parser, data_path, lang_idx=0, direction=0)
        # train_dl, test_dl, val_dl, train_ds, test_ds, val_ds
        # TODO: why return a tuple? turn this into a dict
        model = ModelWrapped(parser, learned_params, ckpt=Checkpoint_saver)
        trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=test_dl)
    if train_left:
        train_dl, test_dl, val_dl, *the_datasets = get_dataset_for_spatial_realtions(
            parser, data_path,  lang_idx=0,    direction=1)
        model = ModelWrapped(parser, learned_params, ckpt=Checkpoint_saver)
        trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=test_dl)


main(True, False)
