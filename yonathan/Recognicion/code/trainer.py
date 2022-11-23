import sys
import pytorch_lightning as pl
from supp.get_dataset import get_dataset_for_spatial_realtions
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import os
from pathlib import Path
from src.Checkpoint_model_definition import CheckpointSaver, ModelWrapped
import torch.nn as nn
from Configs.Config import Config
from supp.create_model import get_or_create_model
import git
# TODO write an own import function using imp
git_repo = git.Repo(__file__, search_parent_directories=True)
git_root = git_repo.working_dir
sys.path.append(str(git_root))


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
        *** MARKED FOR DEPRECATION ***
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


def main(train_right=True, train_left=False):
    git_repo = git.Repo(__file__, search_parent_directories=True)
    git_root = git_repo.working_dir
    try:
        project_path = Path(git_root)
    except Exception:
        project_path = Path(os.getcwd())
    data_path = os.path.join(
        project_path.parent, 'data/1_extended_testing')
    # TODO change these hard coded paths!
    tmpdir = os.path.join(project_path.parent, 'data/emnist/results/')
    checkpoint_path = os.path.join(tmpdir, 'MyFirstCkt.ckpt')
    parser = Config()
    ModelCkpt = ModelCheckpoint(
        dirpath=tmpdir, monitor="train_loss_epoch", mode="min")
    Checkpoint_saver = CheckpointSaver(
        dirpath=checkpoint_path, decreasing=False, top_n=5)
    # TODO: remove this classs since the checkpoint is already defined
    wandb_logger = WandbLogger(project="My_first_project_5.10", job_type='train',
                               save_dir=f'{git_root}/yonathan/Recognicion/data/emnist/results/')

    trainer = pl.Trainer(accelerator='gpu', max_epochs=60,
                         logger=wandb_logger, callbacks=[ModelCkpt])
    training_flag = Training_flag(
        train_all_model=True, train_arg=False, train_task_embedding=False, train_head=False)

    model = get_or_create_model(model_opts=parser)
    learned_params = training_flag.Get_learned_params(
        model, lang_idx=0, direction=0)
    if train_right:
        train_dl, test_dl, val_dl, *the_datasets = get_dataset_for_spatial_realtions(
            parser, data_path, lang_idx=0, direction=0)
        # train_dl, test_dl, val_dl, train_ds, test_ds, val_ds
        # TODO: why return a tuple? turn this into a dict
    if train_left:
        train_dl, test_dl, val_dl, *the_datasets = get_dataset_for_spatial_realtions(
            parser, data_path,  lang_idx=0,    direction=1)
    model_wrapped = ModelWrapped(
        parser, learned_params, ckpt=Checkpoint_saver, model=model, nbatches_train=len(train_dl))
    wandb_logger.watch(model=model_wrapped, log='all')
    # log all model topology and grads tp the website
    trainer.fit(model_wrapped, train_dataloaders=train_dl,
                val_dataloaders=test_dl)


main(True, False)
