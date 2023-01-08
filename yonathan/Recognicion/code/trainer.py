import argparse
import copy
import os
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
from training.Metrics.Accuracy import accuracy
from training.Modules.Batch_norm import load_running_stats
from training.Modules.Create_Models import create_model
from training.Modules.Models import BUTDModel

cmd_parser = argparse.ArgumentParser()
cmd_parser.add_argument('--data_dir',default='16_extended',type =str,help = "The Data_Creation name")
cmd_parser.add_argument('--ds_type', default='Emnist',type = str, help = "Data set type flag")


def define_wrapped_model(parser, training_flag, model,data_path, direction,lang_idx, Checkpoint_saver ):
    learned_params = training_flag.Get_learned_params(model, task_idx=lang_idx, direction=direction)
    Initial_DataLoaders = get_dataset_for_spatial_relations(parser, data_path, lang_idx=lang_idx, direction_tuple=direction)
    wrapped_model = ModelWrapped(parser, model, learned_params, check_point=Checkpoint_saver,
                                 direction_tuple=direction,
                                 task_id=lang_idx,
                                 nbatches_train=len(Initial_DataLoaders['train_dl']))
    return wrapped_model, Initial_DataLoaders


def main(flag=Flag.CL):
    ds_type = cmd_parser.parse_args().ds_type
    if ds_type  == "Emnist":
        Data_type = DsType.Emnist
        Non_initial_tasks:list[tuple[tuple[int,int],int]] = [((0,1),0),((1,1),0),((-1,-1),0),((2,0),0)] # Four additional tasks.
        initial_tasks = [(1,0),(-1,0)]
        All_tasks = [(1,0),(0,1),(1,1),(-1,-1),(2,0)]
    elif ds_type == "Fashionmnist":
        Data_type = DsType.FashionMnist
        Non_initial_tasks = [(-1,0), (0,1),(0,-1)] # Three addtional tasks.
        initial_tasks = [(1,0)]

    parser = GetParser(task_idx=0, direction_idx=0, model_flag=flag, ds_type=Data_type, model_type=BUTDModel)
    project_path = Path(__file__).parents[1]
    results_dir = os.path.join(project_path, 'Data_Creation/{}/results/model'.format(Data_type.Enum_to_name()))
    data_path = os.path.join(project_path, 'Data_Creation/{}/samples/{}'.format(Data_type.Enum_to_name(), cmd_parser.parse_args().data_dir))
    logger_dir = os.path.join(project_path, 'Data_Creation/emnist/results/')
    Model_checkpoint = ModelCheckpoint(dirpath=logger_dir, monitor="val_loss_epoch", mode="min")
    Checkpoint_saver = CheckpointSaver(dirpath = parser.model_dir , store_running_statistics=flag is Flag.CL)
    wandb_logger = WandbLogger(project="My_first_project_5.10", job_type='train',
                               save_dir=results_dir)
    trainer = pl.Trainer(accelerator='gpu', max_epochs=parser.EPOCHS, logger=wandb_logger, callbacks=[Model_checkpoint])

    model = create_model(parser)

    # Train the initial Task:
    initial_direction,initial_task = initial_tasks[0]
    training_flag = Training_flag(parser, train_all_model = True)
    wrapped_model, Initial_DataLoaders = define_wrapped_model(parser, training_flag, model,data_path, initial_direction, initial_task, Checkpoint_saver )
    trainer.fit(wrapped_model, train_dataloaders=Initial_DataLoaders['train_dl'], val_dataloaders=Initial_DataLoaders['test_dl'])
    print("Begin to train on the initial tasks:")
    Old_models = [copy.deepcopy(model)]
    print("Done fitting the model on the initial tasks")
    if (Data_type is DsType.Emnist or Data_type is DsType.FashionMnist):
        for direction, task in Non_initial_tasks:
            training_flag = Training_flag(parser, train_task_embedding=True,  train_head=True) # Train the embedding and the read-out head.
            wrapped_model, DataLoaders = define_wrapped_model(parser, training_flag, model, data_path,
                                                                      direction, task, Checkpoint_saver)
            trainer.fit(wrapped_model, train_dataloaders=DataLoaders['train_dl'], val_dataloaders=DataLoaders['test_dl'])
            trainer = pl.Trainer(accelerator='gpu', max_epochs=parser.EPOCHS, logger=wandb_logger, callbacks=[Model_checkpoint])
            Old_models.append(copy.deepcopy(model))
    print("Done training on the new tasks")
    Acc_old =  []
    Acc_new =  []
    for task_idx, (direction, task) in enumerate(All_tasks):
            _ , DataLoaders = define_wrapped_model(parser, training_flag, model, data_path,
                                                              task, task, Checkpoint_saver)
            Sub_task = All_tasks[task_idx]
            load_running_stats(model, 0, Sub_task)  # Loading the saved running statistics.
            load_running_stats(Old_models[task_idx], 0, Sub_task)  # Loading the saved running statistics.
            acc_old = accuracy(opts=parser, model=Old_models[task_idx], test_data_loader=DataLoaders['test_dl'])
            Acc_old.append(acc_old.item())
            acc_new = accuracy(opts=parser, model=model, test_data_loader=DataLoaders['test_dl'])
            Acc_new.append(acc_new.item())
    for idx, task in enumerate(All_tasks):
        old = Acc_old[idx]
        new = Acc_new[idx]
        print("The Accuracy on the task:{} was: {.3f} and now is: {.3f}".format(task,old,new))
   # print(Acc_new)
  #  print(Acc_old)
  #  print("The Accuracy on the initial task was: {}".format(acc_old) )

   # print("The Accuracy on the initial task now is: {}".format(acc_new))
    print("As all rows the same there is No forgetting!")

main()


