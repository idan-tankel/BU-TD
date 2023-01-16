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
from training.Modules.Batch_norm import BatchNorm
from training.Utils import num_params


def main_fashion(train_original: bool, train_new: bool, ds_type: DsType = DsType.Fashionmnist, flag: Flag = Flag.CL,
                 model_type: nn.Module = BUTDModel, task: tuple = (-1, 0), num_epochs=0) -> None:
    """
    Trains the model.
    Args:
        train_original: Whether to train original direction.
        train_new: Whether train new direction.
        ds_type: The dataset type.
        flag: The Model flag.
        model_type: The model type.
        task: The task.
        num_epochs: The number of epochs.

    """
    parser = GetParser(model_flag=flag, ds_type=ds_type, model_type=model_type)
    project_path = Path(__file__).parents[1]
    results_dir = os.path.join(project_path,
                               f'data/{str(ds_type)}/results/Task_{task}smaller_model_model_wd_{parser.wd}'
                               f'_{num_epochs}_epochs')
    data_path = os.path.join(project_path, 'data/{}/samples/(3,3)_Image_Matrix'.format(str(ds_type)))
    Checkpoint_saver = CheckpointSaver(dirpath=results_dir + f"Model_{task[0]}")
    wandb_logger = WandbLogger(project="My_first_project_5.10", job_type='train',
                               save_dir=results_dir)
    trainer = pl.Trainer(accelerator='gpu', max_epochs=parser.EPOCHS, logger=wandb_logger)

    model = create_model(parser)
    if train_original:
        training_flag = Training_flag(parser, train_all_model=True)
        direction = [(1, 0)]
        # learned_params = list(model.parameters())
        learned_params = training_flag.Get_learned_params(model, task_idx=0, direction=direction[0])
        DataLoaders = get_dataset_for_spatial_relations(parser, data_path, lang_idx=0, direction_tuple=direction)
        wrapped_model = ModelWrapped(parser, model, learned_params, check_point=Checkpoint_saver,
                                     direction_tuple=direction,
                                     task_id=0,
                                     nbatches_train=len(DataLoaders['train_dl']), train_ds=DataLoaders['train_ds'])
        #   wrapped_model.load_model(model_path='Right_model/BUTDModel_latest_direction=(1, 0).pt')
        # print(wrapped_model.Accuracy(DataLoaders['test_dl']))
        trainer.fit(wrapped_model, train_dataloaders=DataLoaders['train_dl'], val_dataloaders=DataLoaders['test_dl'])

    if train_new:
        direction = task  #
        '''
        BN = []
        for layer in model.modules():
            if isinstance(layer, BatchNorm):
                BN.extend(layer.parameters())
        print(num_params(BN))
        '''
        training_flag = Training_flag(parser, train_task_embedding=True, train_head=True)
        learned_params = training_flag.Get_learned_params(model, task_idx=0, direction=direction[0])
        DataLoaders = get_dataset_for_spatial_relations(parser, data_path, lang_idx=0, direction_tuple=direction)
        wrapped_model = ModelWrapped(parser, model, learned_params, check_point=Checkpoint_saver,
                                     direction_tuple=direction,
                                     task_id=0,
                                     nbatches_train=len(DataLoaders['train_dl']), train_ds=DataLoaders['train_ds'])
        wrapped_model.load_model(
            model_path=f'Task_[(0, 1)]smaller_model_model_wd_1e-05_70_epochsModel_(0, 1)/BUTDModel_best_direction=(0, 1).pt')
        print(wrapped_model.Accuracy(DataLoaders['test_dl']))
        print(num_params(learned_params))
        parser.model = model
        trainer.fit(wrapped_model, train_dataloaders=DataLoaders['train_dl'], val_dataloaders=DataLoaders['test_dl'])


main_fashion(False, True, task=[(0, 1)], num_epochs=60)
