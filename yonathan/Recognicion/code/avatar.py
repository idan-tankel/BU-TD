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
from training.Modules.Batch_norm import BatchNorm
from training.Utils import num_params
import torch.nn.modules as Module


def main(train_right, train_left, ds_type:DsType=DsType.Avatar, flag:Flag=Flag.CL, model_type:Module=BUTDModel, task:list=[(-1, 0)],epoch = 0):
    parser = GetParser(model_flag=flag, ds_type=ds_type, model_type=model_type)
    project_path = Path(__file__).parents[1]
    data_path = os.path.join(project_path, f'data/{str(ds_type)}/samples/(2,2)_Test_open_files')
    Checkpoint_saver = CheckpointSaver(dirpath=parser.results_dir + f'Model_non_reset_{str(task[0])}_wd_{str(parser.wd)}_base_lr_{parser.base_lr}_max_lr_{parser.max_lr}_epoch_{epoch}_option_bs_{parser.bs}')
    wandb_logger = WandbLogger(project="My_first_project_5.10", save_dir=parser.results_dir)
    trainer = pl.Trainer(max_epochs=parser.EPOCHS, logger=wandb_logger, accelerator = 'gpu')
    model = create_model(parser)
    if train_right:
        training_flag = Training_flag(parser, train_all_model=True)
        direction = parser.initial_directions
        learned_params = training_flag.Get_learned_params(model, task_idx=0, direction=direction[0])
        DataLoaders = get_dataset_for_spatial_relations(parser, data_path, lang_idx=0, direction_tuple=direction)
        wrapped_model = ModelWrapped(parser, model, learned_params, check_point=Checkpoint_saver,
                                     train_ds=DataLoaders['train_ds'],
                                     direction_tuple=direction,
                                     task_id=0,
                                     nbatches_train=len(DataLoaders['train_dl']))
        '''
        labels = [0 for _ in range(3)]
        for inputs in DataLoaders['train_dl']:
            for j in range(3):
             label_task = inputs[1]
             labels[j] += (label_task == j).sum()

        print(labels)
        '''


     #   check = wrapped_model.load_model(
     #       model_path='Model_(1, 0)_1e-05_test_again/BUTDModel_epoch50_direction=[(1, 0), (-1, 0)].pt')
     #   print(wrapped_model.Accuracy(DataLoaders['test_dl']))
        trainer.fit(wrapped_model, train_dataloaders=DataLoaders['train_dl'], val_dataloaders=DataLoaders['test_dl'])

    if train_left:
        direction = task  #
        # print(num_params(model.parameters()))
        training_flag = Training_flag(parser, train_task_embedding=True, train_head=True)
        learned_params = training_flag.Get_learned_params(model, task_idx=0, direction=task[0])
    #    print(num_params(learned_params))
        DataLoaders = get_dataset_for_spatial_relations(parser, data_path, lang_idx=0, direction_tuple=direction)
        wrapped_model = ModelWrapped(parser, model, learned_params, check_point=Checkpoint_saver,
                                     direction_tuple=direction,
                                     task_id=0,
                                     nbatches_train=len(DataLoaders['train_dl']),train_ds=DataLoaders['train_ds'])
        ''''
        BN =[]
        for layer in model.modules():
            if isinstance(layer, BatchNorm):
                BN.extend(layer.parameters())
        print(num_params(BN))
        '''
        # 44
        wrapped_model.load_model(model_path=f'Model_(1, 0)_single_base/BUTDModel_epoch{epoch}_direction=[(1, 0)].pt', load_opt_and_sche=False)
     #   print(check['epoch'])
     #   acc = wrapped_model.Accuracy(DataLoaders['test_dl'])
       # print("{:.4f}".format(acc))
        trainer.fit(wrapped_model, train_dataloaders=DataLoaders['train_dl'], val_dataloaders=DataLoaders['test_dl'])


main(True, True, task=[(-1,0)],epoch=60)