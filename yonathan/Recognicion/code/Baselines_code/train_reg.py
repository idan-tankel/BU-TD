import os
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

sys.path.append(os.path.join('r', Path(__file__).parents[1]))
from training.Data.Checkpoints import CheckpointSaver
from training.Data.Data_params import Flag, DsType
from training.Data.Get_dataset import get_dataset_for_spatial_relations
from training.Data.Model_Wrapper import ModelWrapped
from training.Data.Parser import GetParser, update_parser
from training.Data.Structs import Training_flag
from training.Data.Data_params import RegType
from training.Modules.Create_Models import create_model
from training.Modules.Models import BUTDModel, ResNet
import torch.nn.modules as Module
from Regulizers.Regulizer import Get_regulizer_according_to_reg_type
from penalty_loss import RegLoss


def main(train_right, train_left, ds_type:DsType=DsType.Emnist, flag:Flag=Flag.CL, model_type:Module=BUTDModel, task:list=[(-1, 0)],reg_type = RegType.LFL):
    parser = GetParser(model_flag=flag, ds_type=ds_type, model_type=model_type)
    project_path = Path(__file__).parents[2]
    data_path = os.path.join(project_path, f'data/{str(ds_type)}/samples/(4,4)_image_matrix')
    Checkpoint_saver = CheckpointSaver(dirpath=parser.baselines_dir+ f'naive/Model_{str(task[0])}_{str(parser.wd)}')
    wandb_logger = WandbLogger(project="My_first_project_5.10", job_type='train',
                               save_dir=parser.results_dir)
    trainer = pl.Trainer(accelerator='gpu', max_epochs=parser.EPOCHS, logger=wandb_logger)

    update_parser(opts=parser, attr='ns', new_value=[3, 3, 3])  # Make the ResNet to be as large as BU-TD model.
    update_parser(opts=parser, attr='use_lateral_bu_td', new_value=False)  # No lateral connections are needed.
    update_parser(opts=parser, attr='use_laterals_td_bu', new_value=False)  # No lateral connections are needed.
    model = create_model(parser)
    model.prev_tasks = [(0,(1,0))]
    if train_right:
        training_flag = Training_flag(parser, train_all_model=True)
        direction = parser.initial_directions
        learned_params = training_flag.Get_learned_params(model, task_idx=0, direction=direction[0])
        DataLoaders = get_dataset_for_spatial_relations(parser, data_path, lang_idx=0, direction_tuple=direction)
        wrapped_model = ModelWrapped(parser, model, learned_params, check_point=Checkpoint_saver,
                                     direction_tuple=direction,
                                     task_id=0,
                                     nbatches_train=len(DataLoaders['train_dl']))
     #   check = wrapped_model.load_model(
     #       model_path='Model_(1, 0)_1e-05_test_again/BUTDModel_epoch50_direction=[(1, 0), (-1, 0)].pt')
     #   print(wrapped_model.Accuracy(DataLoaders['test_dl']))
        trainer.fit(wrapped_model, train_dataloaders=DataLoaders['train_dl'], val_dataloaders=DataLoaders['test_dl'])

    if train_left:
        direction = task  #

        results_path = parser.baselines_dir  # Getting the result dir.
        model_path = os.path.join(results_path, 'naive/Model_right/ResNet_epoch40_direction=(1, 0).pt')  # The path to the model.
        checkpoint = torch.load(model_path)  # Loading the saved data.
        DataLoaders = get_dataset_for_spatial_relations(parser, data_path, lang_idx=0, direction_tuple=direction)
        model.load_state_dict(checkpoint['model_state_dict'])  # Loading the saved weights.
        parser.model = model
        Reg_type = Get_regulizer_according_to_reg_type(parser=parser, reg_type=reg_type, prev_model = model, prev_data = DataLoaders['train_ds'])
        Loss = RegLoss(opts=parser, Reg=Reg_type)
        update_parser(opts=parser, attr='criterion', new_value=Loss)
        # print(num_params(model.parameters()))
        training_flag = Training_flag(parser, train_all_model = True)
        learned_params = training_flag.Get_learned_params(model, task_idx=0, direction=task[0])
       # print(num_params(learned_params))

        wrapped_model = ModelWrapped(parser, model, learned_params, check_point=Checkpoint_saver,
                                     direction_tuple=direction,
                                     task_id=0,
                                     nbatches_train=len(DataLoaders['train_dl']))
        # 44
 #       check = wrapped_model.load_model(model_path=f'Model_{task[0]}_1e-05_test/BUTDModel_best_direction={task}.pt')
     #   print(check['epoch'])
    #    print(wrapped_model.Accuracy(DataLoaders['test_dl']))
   #     print(wrapped_model.Accuracy(DataLoaders['train_dl']))
        trainer.fit(wrapped_model, train_dataloaders=DataLoaders['train_dl'], val_dataloaders=DataLoaders['test_dl'])


main(False, True, task=[(0,1)],reg_type=RegType.LFL,flag=Flag.NOFLAG,model_type=ResNet)