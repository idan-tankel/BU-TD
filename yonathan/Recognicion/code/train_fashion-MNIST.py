"""
Full EMNIST training.
"""
import copy

from training.Data.Parser import GetParser
from training.Data.Structs import Training_flag
from training.Modules.Create_Models import create_model
from training.training_funcs.training_step import train_step
from training.Modules.Models import *


def train_trajectory():
    """
    Returns:

    """
    Models = []
    opts = GetParser(model_flag=Flag.CL, ds_type=DsType.Fashionmnist)
    model = create_model(opts)
    first_task = opts.initial_directions
    training_flag = Training_flag(opts, train_all_model=True)
    new_model = train_step(model_opts=opts, model=model, task=first_task, training_flag=training_flag,
                           ds_type=DsType.Fashionmnist)
    Models.append(new_model)
    new_tasks = [[(-1, 0)], [(0, 1)], [(0, -1)]]
    for task in new_tasks:
        training_flag = Training_flag(opts, train_head=True, train_task_embedding=True)
        model = train_step(model_opts=opts, model=model, task=task, training_flag=training_flag)
        Models.append(copy.deepcopy(model))
    print(len(Models))


train_trajectory()
