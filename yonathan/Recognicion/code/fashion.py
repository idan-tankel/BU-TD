import os
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from training.Data.Checkpoints import CheckpointSaver
from training.Data.Data_params import Flag, DsType
from training.Data.Get_dataset import get_dataset_for_spatial_relations
from training.Data.Model_Wrapper import ModelWrapped
from training.Data.Parser import GetParser, update_parser
from training.Data.Structs import Training_flag
from training.Modules.Create_Models import create_model
from training.Modules.Models import BUTDModel


def main(train_right, train_left, ds_type=DsType.Fashionmnist, flag=Flag.CL, model_type = BUTDModel, task = (-1, 0)):
    parser = GetParser(task_idx=0, direction_idx=0, model_flag=flag, ds_type=ds_type, model_type=model_type)
    project_path = Path(__file__).parents[1]
    results_dir = os.path.join(project_path, 'data/{}/results/model'.format(str(ds_type)))
    data_path = os.path.join(project_path, 'data/{}/samples/(3,3)_Image_Matrix'.format(str(ds_type)))
    tmpdir = os.path.join(project_path, 'data/emnist/results/')
    now = datetime.now()
    time = now.strftime("%m.%d.%Y%H:%M:%S")
    Model_checkpoint = ModelCheckpoint(dirpath=tmpdir, monitor="val_loss_epoch", mode="min")
    Checkpoint_saver = CheckpointSaver(dirpath=results_dir + time, store_running_statistics=flag is Flag.CL)
    wandb_logger = WandbLogger(project="My_first_project_5.10", job_type='train',
                               save_dir=results_dir)
    trainer = pl.Trainer(accelerator='gpu', max_epochs=60, logger=wandb_logger, callbacks=[Model_checkpoint],
                         reload_dataloaders_every_n_epochs=1)

    model = create_model(parser)
    if train_right:
        training_flag = Training_flag(parser, train_all_model=True)
        direction = (1, 0)
       # learned_params = list(model.parameters())
        learned_params = training_flag.Get_learned_params(model, task_idx = 0, direction=direction)
        DataLoaders = get_dataset_for_spatial_relations(parser, data_path, lang_idx=0, direction_tuple=direction)
        wrapped_model = ModelWrapped(parser, model, learned_params, check_point=Checkpoint_saver,
                                     direction_tuple=direction,
                                     task_id=0,
                                     nbatches_train=len(DataLoaders['train_dl']))
     #   wrapped_model.load_model(model_path='Right_model/BUTDModel_latest_direction=(1, 0).pt')
       # print(wrapped_model.Accuracy(DataLoaders['test_dl']))
        trainer.fit(wrapped_model, train_dataloaders=DataLoaders['train_dl'], val_dataloaders=DataLoaders['test_dl'])

    if train_left:
        direction = task  #

        training_flag = Training_flag(parser, train_all_model=False, train_arg=False, train_task_embedding=True,
                                      train_head=True)
        learned_params = training_flag.Get_learned_params(model, task_idx = 0, direction=direction)
        DataLoaders = get_dataset_for_spatial_relations(parser, data_path, lang_idx=0, direction_tuple=direction)
        wrapped_model = ModelWrapped(parser, model, learned_params, check_point=Checkpoint_saver,
                                     direction_tuple=direction,
                                     task_id=0,
                                     nbatches_train=len(DataLoaders['train_dl']))
        wrapped_model.load_model(model_path='Model_(1,0)_new_ver/BUTDModel_epoch49_direction=(1, 0).pt')
        print(wrapped_model.Accuracy(DataLoaders['test_dl']))
        parser.model = model
        trainer.fit(wrapped_model, train_dataloaders=DataLoaders['train_dl'], val_dataloaders=DataLoaders['test_dl'])

main(False, True, ds_type=DsType.Fashionmnist, model_type=BUTDModel, flag=Flag.CL, task = (-1,0))