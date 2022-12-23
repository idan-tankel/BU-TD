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
from training.Data.Parser import GetParser
from training.Data.Structs import Training_flag
from training.Modules.Create_Models import create_model
from training.Modules.Models import BUTDModel, ResNet
from training.Utils import num_params


def main(train_right, train_left, ds_type=DsType.Emnist, flag=Flag.CL, model_type=BUTDModel, task=(-1, 0)):
    parser = GetParser(task_idx=0, direction_idx=0, model_flag=flag, ds_type=ds_type, model_type=model_type)
    project_path = Path(__file__).parents[1]
    data_path = os.path.join(project_path, f'data/{str(ds_type)}/samples/(4,4)_image_matrix')
    Checkpoint_saver = CheckpointSaver(dirpath=parser.results_dir + f'Model{task}_train_all_rows_larger_emb{parser.wd}', store_running_statistics=flag is Flag.CL)
    wandb_logger = WandbLogger(project="My_first_project_5.10", job_type='train',
                               save_dir=parser.results_dir)
    trainer = pl.Trainer(accelerator='gpu', max_epochs=parser.EPOCHS, logger=wandb_logger)
    model = create_model(parser)
    if train_right:
        training_flag = Training_flag(parser, train_all_model=True)
        direction = (1,0)
        learned_params = training_flag.Get_learned_params(model, task_idx=0, direction=direction)
        DataLoaders = get_dataset_for_spatial_relations(parser, data_path, lang_idx=0, direction_tuple=direction)
        wrapped_model = ModelWrapped(parser, model, learned_params, check_point=Checkpoint_saver,
                                     direction_tuple=direction,
                                     task_id=0,
                                     nbatches_train=len(DataLoaders['train_dl']))
        #wrapped_model.load_model(model_path='Model_right/ResNet_epoch14_direction=(1, 0).pt')
    #    print(wrapped_model.Accuracy(DataLoaders['test_dl']))
        trainer.fit(wrapped_model, train_dataloaders=DataLoaders['train_dl'], val_dataloaders=DataLoaders['test_dl'])

    if train_left:
        direction = task  #
        # print(num_params(model.parameters()))
        training_flag = Training_flag(parser, train_task_embedding=True, train_head=True)
        learned_params = training_flag.Get_learned_params(model, task_idx=0, direction=task)
        print(num_params(learned_params))
        DataLoaders = get_dataset_for_spatial_relations(parser, data_path, lang_idx=0, direction_tuple=direction)
        wrapped_model = ModelWrapped(parser, model, learned_params, check_point=Checkpoint_saver,
                                     direction_tuple=direction,
                                     task_id=0,
                                     nbatches_train=len(DataLoaders['train_dl']))
        wrapped_model.load_model(model_path='Right_larger_emb/BUTDModel_epoch30_direction=(1, 0).pt')
     #   print(wrapped_model.Accuracy(DataLoaders['test_dl']))
        trainer.fit(wrapped_model, train_dataloaders=DataLoaders['train_dl'], val_dataloaders=DataLoaders['test_dl'])


main(True, True, ds_type=DsType.Emnist, model_type=BUTDModel, flag=Flag.CL, task=(-2, 0))
