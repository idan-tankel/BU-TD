import os
import sys
from pathlib import Path

sys.path.append(os.path.join('r', Path(__file__).parents[1]))
from Baselines_code.avalanche_AI.training.Plugins.classes import RegType

from training.Data.Get_dataset import get_dataset_for_spatial_relations
from avalanche.benchmarks.generators import dataset_benchmark
from avalanche_AI.training.supervised.strategy_wrappers import Regularization_strategy
from training.Metrics.Accuracy import accuracy
from Baselines_code.baselines_utils import Get_updated_opts, Get_samples_data
from training.Data.Structs import Task_to_struct
from training.Utils import *
from baselines_utils import load_model
from typing import Union
from training.Modules.Models import *


def main(reg_type: Union[RegType, None], ds_type: DsType, new_task, load):
    """
    Args:
        reg_type: The regularization type.
        ds_type: The data-set type.
        new_task: The new list_task_structs.
        load: Whether to load pretrained model_test.

    Returns:

    """
    # The model_test model_opts: NO-FLAG mode, ResNet model_test.
    checkpoint = None
    opts, model = Get_updated_opts(ds_type=ds_type, reg_type=reg_type,model_type=BUTDModel, model_flag=Flag.CL)
    if reg_type is not RegType.SI:
        path = 'Smaller_model_Task_[0, (1, 0)]/Naive/lambda=0/ResNet_epoch40_direction=(1, 0).pt'
        #  path = 'Task_[0, (0, 1)]/LWF/lambda=0.1/ResNet_epoch10_direction=(0, 1).pt'
        #  path = 'Smaller_model_Task_[0, (-2, 0)]/LWF/lambda=0.03/ResNet_epoch15_direction=(-2, 0).pt'
        #   path = 'Smaller_model_Task_[0, (0, 1)]/Naive/lambda=0/ResNet_epoch40_direction=(0, 1).pt'
        #  path = 'Task_[0, (0, 1)]/LFL/lambda=0.35/ResNet_epoch10_direction=(0, 1).pt'
        #  path = 'Smaller_model_Task_[0, (-2, 0)]/LFL/lambda=0.15/ResNet_epoch14_direction=(-2, 0).pt'
        #   path = 'Smaller_model_Task_[0, (-2, 0)]/LWF/lambda=0.029/ResNet_epoch9_direction=(-2, 0).pt'
        #  model_path = 'Smaller_model_Task_[0, (0, 1)]/EWC/lambda=0.98/ResNet_epoch6_direction=(0, 1).pt'
        #   path = 'Smaller_model_Task_[0, (0, 1)]/LWF/lambda=0.4/ResNet_epoch16_direction=(0, 1).pt'
        #   path = 'Smaller_model_Task_[0, (-1, -1)]/LWF/lambda=0.3/ResNet_epoch16_direction=(-1, -1).pt'
        path = 'Smaller_model_Task_[0, (0, 1)]/LWF/lambda=0.1/ResNet_epoch23_direction=(0, 1).pt'
        path = 'Smaller_model_Task_[0, (1, 1)]/LWF/lambda=0.1/ResNet_epoch40_direction=(1, 1).pt'
        path = 'Smaller_model_Task_[0, (1, -1)]/LWF/lambda=0.08/ResNet_epoch18_direction=(1, -1).pt'
        path = 'Smaller_model_Task_(0, (-1, 0))/LWF/lambda=0.45/ResNet_epoch30_direction=(-1, 0).pt'
        path = 'Right_model/BUTDModel_epoch70_direction=(1, 0).pt'
    else:
        path = 'Smaller_model_Task_[0, (1, 0)]/SI/lambda=0.125/ResNet_epoch40_direction=(1, 0).pt'
    model_path = os.path.join(opts.results_dir, path)
    if load:
        checkpoint = load_model(model, model_path=model_path,
                                results_path=opts.baselines_dir)
    opts.model.trained_tasks.append((0, (1, 0)))
    # Path to the data-set.
    Data_specific_path = opts.Data_specific_path
    # Path to the samples.
    sample_path = Get_samples_data(ds_type=ds_type)
    # The path to the samples.
    opts.Images_path = os.path.join(Data_specific_path, sample_path)
    # The new tasks.
    task = [Task_to_struct(task = 0, direction = new_task[1])]
    new_data = get_dataset_for_spatial_relations(opts = opts,data_fname = opts.Images_path, list_task_structs = task)
    task = [Task_to_struct(task=0, direction=(1,0))]
    old_data = get_dataset_for_spatial_relations(opts, opts.Images_path, list_task_structs=task)
    test = False
    if test:
        model_path = 'Smaller_model_Task_(0, (-1, -1))/LWF/lambda=0.4/ResNet_epoch17_direction=(-1, -1).pt'
        #   model_path =  'Smaller_model_Task_[0, (1, 0)]/Naive/lambda=0/ResNet_epoch40_direction=(1, 0).pt'
        checkpoint = load_model(model, model_path=model_path,
                                results_path=opts.baselines_dir)
        print(accuracy(opts, model, new_data['test_dl']))
        print(accuracy(opts, model, old_data['test_dl']))
    # The scenario.
    #  model_path = os.path.join(model_opts.baselines_dir, f'{new_task}/{str(reg_type)}/{reg_type.class_to_reg_factor(model_opts)}')
    scenario = dataset_benchmark([new_data['train_ds']], [new_data['test_ds']])
    # Train the baselines.
    strategy = Regularization_strategy(opts, eval_every=1, prev_model=checkpoint,
                                       model_path=model_path, task=new_task, reg_type=reg_type)
    strategy.train_sequence(scenario)


main(reg_type=RegType.LWF, ds_type=DsType.Omniglot, new_task=(50, (-1,0)), load=True,)
