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


def train_trajectory():
    """
    Returns:

    """
    Models = []
    opts = GetParser(model_flag=Flag.CL, ds_type=DsType.Omniglot)
    model = create_model(opts)
    first_task = [Task_to_struct(task=50, direction=(1, 0))]
    training_flag = Training_flag(opts, train_all_model=True)
    new_model = train_step(model_opts=opts, model=model, task=first_task, training_flag=training_flag,
                           ds_type=DsType.Omniglot)
    Models.append(new_model)
    new_tasks = [[1, (1, 0)], [2, (1, 0)], [3, (1, 0)]]
    for direction, lang_id in new_tasks:
        training_flag = Training_flag(opts, train_head=True, train_task_embedding=True)
        model = train_step(model_opts=opts, model=model, task=direction, training_flag=training_flag,
                           ds_type=DsType.Omniglot)
        Models.append(copy.deepcopy(model))

    print(len(Models))


train_trajectory()
