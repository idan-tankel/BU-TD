import sys
sys.path.append(r'/home/sverkip/data/BU-TD/yonathan/Recognicion/code/')
import pytorch_lightning as pl
from supp.Parser import GetParser
from supp.get_dataset import get_dataset_for_spatial_realtions
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import os
from pathlib import Path
from supp.pytorch_lightning_model_and_checkpoints import CheckpointSaver, ModelWrapped, Training_flag
from supp.Dataset_and_model_type_specification import Flag

def main(train_right, train_left):
    parser = GetParser(language_idx=0, direction='right',flag=Flag.ZF)
    project_path = Path(__file__).parents[1]
    data_path = os.path.join(project_path, 'data/emnist/samples/6_extended')
    tmpdir = os.path.join(project_path, 'data/emnist/results/')
    checkpoint_path = os.path.join(tmpdir, 'MyFirstCkt.ckpt')
    ModelCkpt = ModelCheckpoint(dirpath=checkpoint_path, monitor="val_loss_epoch", mode="min")
    Checkpoint_saver = CheckpointSaver( dirpath='/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/results/MyFirstCkpt', decreasing=False, top_n=5)
    wandb_logger = WandbLogger(project="My_first_project_5.10", job_type='train',
                               save_dir='/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/results/')
    trainer = pl.Trainer(accelerator='gpu', max_epochs=60, logger=wandb_logger, callbacks=[ModelCkpt])
    training_flag = Training_flag(parser, train_all_model = True, train_arg = False, train_task_embedding = False, train_head = False)
    learned_params = training_flag.Get_learned_params(parser.model, lang_idx = 0, direction = 0)
    if train_right:
        [train_dl, test_dl, val_dl, _, _, _] = get_dataset_for_spatial_realtions(parser, data_path, lang_idx=0, direction = 0)
        model = ModelWrapped(parser, learned_params, ckpt=Checkpoint_saver,direction = 0)
        trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=test_dl)
    if train_left:
        [the_datasets, train_dl, test_dl, val_dl, _, _, _] = get_dataset_for_spatial_realtions(parser, data_path,  lang_idx=0,    direction = 1)
        model = ModelWrapped(parser, learned_params, ckpt=Checkpoint_saver)
        trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=test_dl)


main(True,False)