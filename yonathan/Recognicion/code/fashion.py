import os

from pathlib import Path

import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.loggers import WandbLogger

from training.Data.Checkpoints import CheckpointSaver
from training.Data.Data_params import Flag, DsType
from training.Data.Get_dataset import get_dataset_for_spatial_relations
from training.Data.Model_Wrapper import ModelWrapped
from training.Data.Parser import GetParser
from training.Data.Structs import Training_flag
from training.Modules.Create_Models import create_model
from training.Modules.Models import BUTDModel


def main_fashion(train_original: bool, train_new: bool, ds_type: DsType = DsType.Fashionmnist, flag: Flag = Flag.CL,
                 model_type: nn.Module = BUTDModel, task: tuple = (-1, 0)) -> None:
    """
    Trains the model.
    Args:
        train_original: Whether to train original direction.
        train_new: Whether train new direction.
        ds_type: The dataset type.
        flag: The Model flag.
        model_type: The model type.
        task: The task.

    """
    parser = GetParser(model_flag=flag, ds_type=ds_type, model_type=model_type)
    project_path = Path(__file__).parents[1]
    results_dir = os.path.join(project_path, 'data/{}/results/model'.format(str(ds_type)))
    data_path = os.path.join(project_path, 'data/{}/samples/(3,3)_Image_Matrix'.format(str(ds_type)))
    Checkpoint_saver = CheckpointSaver(dirpath=results_dir + f"Model{task}", store_running_statistics=flag is Flag.CL)
    wandb_logger = WandbLogger(project="My_first_project_5.10", job_type='train',
                               save_dir=results_dir)
    trainer = pl.Trainer(accelerator='gpu', max_epochs=parser.EPOCHS, logger=wandb_logger)

    model = create_model(parser)
    if train_original:
        training_flag = Training_flag(parser, train_all_model=True)
        direction = (1, 0)
        # learned_params = list(model.parameters())
        learned_params = training_flag.Get_learned_params(model, task_idx=0, direction=direction)
        DataLoaders = get_dataset_for_spatial_relations(parser, data_path, lang_idx=0, direction_tuple=direction)
        wrapped_model = ModelWrapped(parser, model, learned_params, check_point=Checkpoint_saver,
                                     direction_tuple=direction,
                                     task_id=0,
                                     nbatches_train=len(DataLoaders['train_dl']))
        #   wrapped_model.load_model(model_path='Right_model/BUTDModel_latest_direction=(1, 0).pt')
        # print(wrapped_model.Accuracy(DataLoaders['test_dl']))
        trainer.fit(wrapped_model, train_dataloaders=DataLoaders['train_dl'], val_dataloaders=DataLoaders['test_dl'])

    if train_new:
        direction = task  #

        training_flag = Training_flag(parser, train_task_embedding=True, train_head=True)
        learned_params = training_flag.Get_learned_params(model, task_idx=0, direction=direction)
        DataLoaders = get_dataset_for_spatial_relations(parser, data_path, lang_idx=0, direction_tuple=direction)
        wrapped_model = ModelWrapped(parser, model, learned_params, check_point=Checkpoint_saver,
                                     direction_tuple=direction,
                                     task_id=0,
                                     nbatches_train=len(DataLoaders['train_dl']))
        wrapped_model.load_model(model_path='Model_(1,0)_new_ver/BUTDModel_epoch49_direction=(1, 0).pt')
        #     print(wrapped_model.Accuracy(DataLoaders['test_dl']))
        parser.model = model
        trainer.fit(wrapped_model, train_dataloaders=DataLoaders['train_dl'], val_dataloaders=DataLoaders['test_dl'])


main_fashion(False, True, task=(-1, 1))
