import os
import sys
from pathlib import Path

sys.path.append(os.path.join('r', Path(__file__).parents[1]))

from training.Data.Data_params import RegType
from training.Data.Parser import GetParser, update_parser
from training.Modules.Create_Models import create_model

from training.Data.Get_dataset import get_dataset_for_spatial_relations
from training.Modules.Models import *
from avalanche.benchmarks.generators import dataset_benchmark
from avalanche_AI.training.supervised.strategy_wrappers import Regularization_strategy
from Data_Creation.Create_dataset_classes import DsType
from training.Metrics.Accuracy import accuracy

from baselines_utils import load_model
from typing import Union


def main(reg_type: Union[RegType, None], ds_type: DsType):
    """
    Args:
        reg_type: The regularization type.
        ds_type: The data-set type.

    Returns:

    """
    # The parser: NO-FLAG mode, ResNet model.
    parser = GetParser(model_type=ResNet, ds_type=ds_type)
    update_parser(opts=parser, attr='ns', new_value=[3, 3, 3])  # Make the ResNet to be as large as BU-TD model.
    update_parser(opts=parser, attr='use_lateral_bu_td', new_value=False)  # No lateral connections are needed.
    update_parser(opts=parser, attr='use_laterals_td_bu', new_value=False)  # No lateral connections are needed.
    update_parser(opts=parser, attr='reg_type', new_value=reg_type)
    model = create_model(parser)  # Create the model.
    model_path = os.path.join(parser.baselines_dir, 'naive/Model_right/ResNet_epoch40_direction=(1, 0).pt')
    load_model(model, model_path='naive/Model_right/ResNet_epoch40_direction=(1, 0).pt',
               results_path=parser.baselines_dir)

    parser.model = model
    parser.model.trained_tasks.append((0, (1, 0)))
    # Path to the project.
    project_path = Path(__file__).parents[2]
    # Path to the data-set.
    Data_specific_path = os.path.join(project_path, 'data/{}'.format(str(ds_type)))
    # Path to the samples.
    if ds_type is DsType.Emnist:
        sample_path = 'samples/(4,4)_image_matrix'
    elif ds_type is DsType.Fashionmnist:
        sample_path = 'samples/(3,3)_Image_Matrix'
    # The path to the samples.
    task = (0, 1)
    Images_path = os.path.join(Data_specific_path, sample_path)
    # The new tasks.
    new_data = get_dataset_for_spatial_relations(parser, Images_path, 0, [task])
    #
    old_data = get_dataset_for_spatial_relations(parser, Images_path, 0, [(1, 0)])
    test = False
    if test:
        load_model(model, model_path=f'LWF/Task_(0, 1)/lambda=4000000/ResNet_epoch5_direction=(1, 0).pt',
                   results_path=parser.baselines_dir)
        print(accuracy(parser, model, new_data['test_dl']))
        print(accuracy(parser, model, old_data['test_dl']))
    #
    new_data = get_dataset_for_spatial_relations(parser, Images_path, 0, [task])
    # The scenario.
    scenario = dataset_benchmark([new_data['train_ds']], [new_data['test_ds']])
    # Train the baselines.
    strategy = Regularization_strategy(parser, eval_every=1, prev_model=parser.model,
                                       model_path=model_path,prev_data = old_data)
    for idx, (exp_train, exp_test) in enumerate(zip(scenario.train_stream, scenario.test_stream)):
        strategy.optimizer.zero_grad(set_to_none=True)  # Make the taskhead of previous task static.
        strategy.train(exp_train, [exp_test], kargs=([0, task], len(exp_train.dataset)))
        print("Done training, now final evaluation of the model.")
        strategy.eval(exp_test)


main(reg_type=RegType.EWC, ds_type=DsType.Emnist)
