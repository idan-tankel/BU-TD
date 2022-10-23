from supp.Dataset_and_model_type_specification import Flag, DsType
from supp.pytorch_lightning_model_and_checkpoints import CheckpointSaver, ModelWrapped, Training_flag
from pathlib import Path
from supp.models import BUTDModelShared, ResNet
import os
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from supp.get_dataset import get_dataset_for_spatial_realtions
from supp.Parser import GetParser
import pytorch_lightning as pl
import git
import sys
sys.path.append(r'/home/sverkip/data/BU-TD/yonathan/Recognicion/code/')

git_repo = git.Repo(os.getcwd(), search_parent_directories=True)
git_root = git_repo.working_dir


def main(train_right, train_left, ds_type=DsType.Emnist):
    parser = GetParser(task_idx=0, direction_idx='right',
                       flag=Flag.NOFLAG, ds_type=ds_type, model_type=ResNet)
    project_path = Path(__file__).parents[1]
    data_path = os.path.join(
        project_path, f'data/{ds_type.Enum_to_name()}/samples/24_extended_testing')
    tmpdir = os.path.join(project_path, 'data/emnist/results/')
    checkpoint_path = os.path.join(tmpdir, 'MyFirstCkt.ckpt')
    Checkpoint_saver = None
    ModelCkpt = ModelCheckpoint(
        dirpath=checkpoint_path, monitor="val_loss_epoch", mode="min")
    wandb_logger = WandbLogger(project="My_first_project_5.10", job_type='train',
                               save_dir='/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/results/')
    trainer = pl.Trainer(accelerator='gpu', max_epochs=60,
                         logger=wandb_logger, callbacks=[ModelCkpt], reload_dataloaders_every_n_epochs=True)
    training_flag = Training_flag(
        parser, train_all_model=True, train_arg=False, train_task_embedding=False, train_head=False)
    learned_params = training_flag.Get_learned_params(
        parser.model, lang_idx=0, direction=0)
    if train_right:
        train_dl, test_dl, val_dl, *datasets = get_dataset_for_spatial_realtions(
            parser, data_path, lang_idx=0, direction=0)
        model = ModelWrapped(opts=parser, learned_params=learned_params,
                             ckpt=Checkpoint_saver, direction=0)
        trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=test_dl)
    if train_left:
        train_dl, test_dl, val_dl, *datasets = get_dataset_for_spatial_realtions(
            parser, data_path,  lang_idx=0,    direction=1)
        model = ModelWrapped(
            opts=parser, learned_params=learned_params, ckpt=Checkpoint_saver)
        trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=test_dl)


if __name__ == '__main__':
    main(train_right=True, train_left=False)
