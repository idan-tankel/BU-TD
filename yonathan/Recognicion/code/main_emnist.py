import os

from pathlib import Path

import torch

import pytorch_lightning as pl

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
from training.Modules.Batch_norm import BatchNormAllTasks


def main(train_right, train_left, ds_type=DsType.Emnist, flag=Flag.CL, model_type=BUTDModel, task=(-1, 0)):
    parser = GetParser(model_flag=flag, ds_type=ds_type, model_type=model_type)

    #  norm = BatchNormAllTasks(opts=parser, num_features=256)
    #  norm.eval()
    #  inputs = torch.ones([10, 256, 100, 100])
    #  flag = torch.zeros(10, 25)
    # flag[0][0] = 1
    #   flag[1][0] = 2
    #   layer = norm(inputs, flag)
    #   print(layer)

    project_path = Path(__file__).parents[1]
    data_path = os.path.join(project_path, f'data/{str(ds_type)}/samples/(4,4)_image_matrix')
    Checkpoint_saver = CheckpointSaver(dirpath=parser.results_dir + f'Model_right_lareger',
                                       store_running_statistics=flag is Flag.CL)
    wandb_logger = WandbLogger(project="My_first_project_5.10", job_type='train',
                               save_dir=parser.results_dir)
    trainer = pl.Trainer(accelerator='gpu', max_epochs=parser.EPOCHS, logger=wandb_logger)
    model = create_model(parser)
    if train_right:
        training_flag = Training_flag(parser, train_all_model=True)
        direction = parser.initial_directions
        learned_params = training_flag.Get_learned_params(model, task_idx=0, direction=direction[0])
        DataLoaders = get_dataset_for_spatial_relations(parser, data_path, lang_idx=0, direction_tuple=direction)
        wrapped_model = ModelWrapped(parser, model, learned_params, check_point=Checkpoint_saver,
                                     direction_tuple=direction,
                                     task_id=0,
                                     nbatches_train=len(DataLoaders['train_dl']))
        #  wrapped_model.load_model(model_path='Model(-2, 0)_train_all_rows_larger_emb1e-05/BUTDModel_epoch11_direction=(1, 0).pt')
        #  print(wrapped_model.Accuracy(DataLoaders['test_dl']))
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
        #   wrapped_model.load_model(model_path='Right_larger_emb/BUTDModel_epoch30_direction=(1, 0).pt')
        #   print(wrapped_model.Accuracy(DataLoaders['test_dl']))
        trainer.fit(wrapped_model, train_dataloaders=DataLoaders['train_dl'], val_dataloaders=DataLoaders['test_dl'])


main(True, True, task=(-2, 0))
