import sys
import os
from pathlib import Path

sys.path.append(os.path.join('r', Path(__file__).parents[1]))
from training.Data.Checkpoints import CheckpointSaver
from training.Data.Data_params import RegType
from training.Data.Parser import GetParser, update_parser
from training.Modules.Create_Models import create_model
from training.Utils import create_optimizer_and_scheduler
from training.Data.Get_dataset import get_dataset_for_spatial_relations
from training.Modules.Models import ResNet, BUTDModel
from avalanche.evaluation.metrics import loss_metrics
from avalanche.logging import WandBLogger, InteractiveLogger
from avalanche.training.plugins.evaluation import EvaluationPlugin
from avalanche.benchmarks.generators import dataset_benchmark

from avalanche_AI.training.Plugins.Evaluation import accuracy_metrics
from avalanche_AI.training.supervised.strategy_wrappers import MyEWC as ewc, LFL, LWF, MyMAS as mas, MyRWALK as rwalk, \
    SI, Naive
from training.Metrics.Accuracy import accuracy
from Data_Creation.Create_dataset_classes import DsType
import argparse
from training.Data.Data_params import Flag
from baselines_utils import load_model
from typing import Union

def train_baseline(parser: argparse, checkpoint: CheckpointSaver, reg_type: RegType, scenario: dataset_benchmark,
                   old_dataset_dict: dict, old_tasks: tuple, new_task: list, Baseline_folder,new_data:dict) -> None:
    """
    Args:
        parser: The parser.
        checkpoint: The checkpoint saver.
        reg_type: The regularization type.
        scenario: The scenario data.
        old_dataset_dict: The old data dictionary.
        old_tasks: The old tasks.
        new_task: The new tasks.

    """
    update_parser(parser=parser, attr='ns', new_value=[0, 3, 3, 3])  # Make the ResNet to be as large as BU-TD model.
    update_parser(parser=parser, attr='use_lateral_bu_td', new_value=False)  # No lateral connections are needed.
    update_parser(parser=parser, attr='use_laterals_td_bu', new_value=False)  # No lateral connections are needed.
    model = create_model(parser)  # Create the model.
    parser.model = model
    parser.model.trained_tasks.append(old_tasks)
    parser.prev_data = old_dataset_dict['train_ds']
    load_model(model,model_path='naive/Model_right/ResNet_epoch40_direction=(1, 0).pt', results_path=Baseline_folder)
  #  print(accuracy(parser, model, old_dataset_dict['test_dl']))
    learned_params = model.parameters() # model.get_specific_head(new_task[0], new_task[1])  # Train only the desired params.
    parser.optimizer, parser.scheduler = create_optimizer_and_scheduler(parser, learned_params, nbatches_train=len(
        old_dataset_dict['train_dl']))
    loggers = [WandBLogger(project_name="avalanche_EWC", run_name="train",
                           dir='/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/results',
                           path='/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/results/checkpoint'),
               InteractiveLogger()]
    evaluator = EvaluationPlugin(accuracy_metrics(parser, minibatch=True, epoch=True, experience=True, stream=True),
                                 loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
                                 benchmark=scenario, loggers=loggers)

    if reg_type is RegType.EWC:
        strategy = ewc(parser, checkpoint=checkpoint, device='cuda', evaluator=evaluator, eval_every=1,
                       prev_model=parser.model)

    elif reg_type is RegType.LWF:
        strategy = LWF(parser, checkpoint=checkpoint, evaluator=evaluator, eval_every=1, prev_model=parser.model)

    elif reg_type is RegType.LFL:
        strategy = LFL(parser, checkpoint=checkpoint, evaluator=evaluator, eval_every=1, prev_model=parser.model)

    elif reg_type is RegType.MAS:
        strategy = mas(parser, checkpoint=checkpoint, device='cuda', evaluator=evaluator, eval_every=1,
                       prev_model=parser.model)

    elif reg_type is RegType.RWALK:
        strategy = rwalk(parser, checkpoint=checkpoint, device='cuda', evaluator=evaluator, train_epochs=1,
                         eval_every=1, prev_model=parser.model)

    elif reg_type is RegType.SI:
        head_params = [key for (key, _) in parser.model.named_parameters() if 'head' in key]
        strategy = SI(parser, checkpoint=checkpoint, device='cuda', evaluator=evaluator, eval_every=1, exc=head_params)
    else:
        strategy = Naive(parser, checkpoint=checkpoint,  evaluator=evaluator, eval_every=1)

    for idx, (exp_train, exp_test) in enumerate(zip(scenario.train_stream, scenario.test_stream)):
        strategy.optimizer.zero_grad(set_to_none=True)  # Make the taskhead of previous task static.
        strategy.train(exp_train, [exp_test])
        print("Done training, now final evaluation of the model.")
        strategy.eval(exp_test)


def main(reg_type: Union[RegType,None], ds_type: DsType):
    """
    Args:
        reg_type: The regularization type.
        ds_type: The data-set type.

    Returns:

    """
    # The parser: NO-FLAG mode, ResNet model.
    parser = GetParser(direction_idx=0, model_type=ResNet, model_flag=Flag.NOFLAG, ds_type=ds_type)
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
    Images_path = os.path.join(Data_specific_path, sample_path)
    # Path to the results.
    results_path = os.path.join(Data_specific_path, 'Baselines')
    # Path to the regularization type results.
 #   Baseline_folder = os.path.join(results_path, str(reg_type))
    # The model folder.
    try:
        Model_folder = os.path.join(results_path, f"{str(reg_type)}/lambda=" + str(reg_type.class_to_reg_factor(parser)))
    except AttributeError:
        Model_folder = os.path.join(results_path, "naive/Model_right" )
    # The checkpoint path.
    checkpoint = CheckpointSaver(Model_folder)

    # The first task is right and then other tasks.
    old_data = get_dataset_for_spatial_relations(parser, Images_path, 0, (1, 0))
    # The new tasks.
    new_data = get_dataset_for_spatial_relations(parser, Images_path, 0, (0,1))
    # The scenario.
    scenario = dataset_benchmark([new_data['train_ds']], [new_data['test_ds']])
    # Train the baselines.
    train_baseline(parser=parser, checkpoint=checkpoint, reg_type=reg_type, scenario=scenario,
                   old_dataset_dict=old_data, old_tasks=(0, (1, 0)), new_task=[0, (0,1)],Baseline_folder = results_path,new_data = new_data)

main(reg_type=RegType.LFL, ds_type= DsType.Emnist)
