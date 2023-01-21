"""
Full EMNIST training.
"""
import copy

from training.Data.Parser import GetParser
from training.Data.Structs import Task_to_struct
from training.Data.Structs import Training_flag
from training.Modules.Create_Models import create_model
from training.Modules.Models import *
from training.training_funcs.training_step import train_step
from training.Utils import load_model


def train_trajectory():
    """
    Returns:

    """
    Models = []
    opts = GetParser(model_flag=Flag.Read_argument, ds_type=DsType.Omniglot,model_type=ResNet)
    opts.use_embedding = True
    model = create_model(opts)
  #  load_model(model=model,results_dir=opts.results_dir,model_path='Model_use_reset_(1,
    #  0)_wd_1e-05_base_lr_0.0002_max_lr_0.002_epoch_0_option_bs_10_use_emb_True' \
    #           '/BUTDModel_epoch15.pt')
    first_task = [Task_to_struct(task=50, direction=(-1, 0))]
    training_flag = Training_flag(opts, train_all_model=True)
    new_model = train_step(opts=opts, model=model, task=first_task, training_flag=training_flag,
                           ds_type=DsType.Omniglot)
    Models.append(new_model)
    new_tasks = [[1, (1, 0)], [2, (1, 0)], [3, (1, 0)]]
    for direction, lang_id in new_tasks:
        training_flag = Training_flag(opts, train_head=True, train_task_embedding=True)
        model = train_step(opts=opts, model=model, task=direction, training_flag=training_flag,
                           ds_type=DsType.Omniglot)
        Models.append(copy.deepcopy(model))

    print(len(Models))


train_trajectory()
