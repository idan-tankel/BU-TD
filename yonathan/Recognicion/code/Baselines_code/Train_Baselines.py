import copy
import os
import sys
from datetime import datetime
from pathlib import Path
# TODO  - GET RID OF THIS.
sys.path.append(r'/home/sverkip/data/BU-TD/yonathan/Recognicion/code/')
from avalanche.benchmarks.generators.benchmark_generators import nc_benchmark
from avalanche.evaluation.metrics import loss_metrics
from avalanche.logging import WandBLogger, InteractiveLogger

from avalanche.training.plugins.evaluation import EvaluationPlugin
from training.Data.Checkpoints import CheckpointSaver
from classes import RegType
from training.Data.Parser import GetParser, update_parser
from training.Data.Structs import Training_flag
from avalanche.benchmarks.generators import dataset_benchmark
from training.Modules.Create_Models import create_model
from training.Utils import create_optimizer_and_scheduler

from training.Data.Get_dataset import get_dataset_for_spatial_relations
from training.Modules.Models import ResNet, BUTDModel
from avalanche_AI.training.Plugins.Evaluation import accuracy_metrics
from avalanche_AI.training.supervised.strategy_wrappers import MyEWC as EWC, LFL, LWF, MyMAS as MAS, MyRWALK as RWALK,SI
from training.Metrics.Accuracy import accuracy
from Data_Creation.Create_dataset_classes import DsType

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def train_baseline(parser,checkpoint, reg_type, scenario, old_dataset_dict, old_tasks):

    update_parser(parser = parser,attr = 'ns',new_value = [0,3,3,3]) # Make the ResNet to be as large as BU-TD model.
    update_parser(parser = parser, attr = 'use_lateral_bu_td',new_value = False) # No lateral connections are needed.
    update_parser(parser=parser, attr='use_laterals_td_bu', new_value=False) # No lateral connections are needed.
    model = create_model(parser) # Create the model.
    parser.model = model
    parser.model.trained_tasks.add(old_tasks)
    parser.prev_data = old_dataset_dict['train_ds']

    learned_params = model.get_specific_head(scenario.task, scenario.direction) # Train only the desired params.
    parser.optimizer, parser.scheduler = create_optimizer_and_scheduler(parser, learned_params, nbatches_train = len(old_dataset_dict['train_dl']))
    loggers = [WandBLogger(project_name="avalanche_EWC", run_name="train",dir='/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/results',path = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/results/checkpoint' ), InteractiveLogger()]

    evaluator = EvaluationPlugin(accuracy_metrics(parser,minibatch=True, epoch=True, experience=True, stream=True),loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),benchmark = scenario,loggers=loggers )

    if reg_type is RegType.EWC:
        strategy =  EWC(parser, checkpoint = checkpoint, device = 'cuda',evaluator = evaluator, eval_every=1,prev_model=parser.model)

    elif reg_type is RegType.LWF:
        strategy = LWF(parser,checkpoint = checkpoint, evaluator=evaluator, eval_every=1,prev_model=parser.model)

    elif reg_type is RegType.LFL:
        strategy = LFL(parser,checkpoint = checkpoint, evaluator=evaluator, eval_every=1,prev_model=parser.model)

    elif reg_type is RegType.MAS:
        strategy = MAS(parser,checkpoint = checkpoint, device='cuda', evaluator=evaluator, eval_every=1,prev_model=parser.model)

    elif reg_type is RegType.RWALK:
        strategy = RWALK(parser,checkpoint = checkpoint, device='cuda', evaluator=evaluator, train_epochs=1, eval_every = 1,prev_model = parser.model)

    elif reg_type is RegType.SI:
        head_params = [key for (key,_) in parser.model.named_parameters() if 'head' in key]
        strategy = SI(parser,checkpoint = checkpoint, device='cuda', evaluator=evaluator, eval_every=1,exc = head_params)
    else:
        raise NotImplementedError

    for idx, (exp_train, exp_test) in enumerate(zip(scenario.train_stream, scenario.test_stream)):
     strategy.optimizer.zero_grad(set_to_none=True) # Make the taskhead of previous task static.
     strategy.train(exp_train, [exp_test])
     print("Done training, now final evaluation of the model.")
     strategy.eval(exp_test)

def main(reg_type, ds_type):
    parser = GetParser(task_idx=0, direction_idx=0, model_type=ResNet)
    project_path = Path(__file__).parents[2]  # Path to the project.
    Data_specific_path = os.path.join(project_path, 'data/{}'.format(str(ds_type)))  # Path to emnist.
    Images_path =  os.path.join(Data_specific_path, 'samples/(4,4)_data_set_matrix_test_changes2') # The path to the data.
    results_path = os.path.join(Images_path, 'results')
    Baseline_folder = os.path.join(results_path, str(reg_type))
    Model_folder = os.path.join(Baseline_folder, "lambda=" + str(reg_type.class_to_reg_factor(parser)))
    checkpoint = CheckpointSaver(Model_folder)
    new_data = get_dataset_for_spatial_relations(parser, Images_path, 0, (0,1))
    old_data = get_dataset_for_spatial_relations(parser, Images_path, 0, (1,0))
    scenario = dataset_benchmark([new_data['train_dl']], [new_data['test_dl']])
    train_baseline(parser=parser, checkpoint=checkpoint, reg_type=reg_type,scenario=scenario,old_dataset_dict=old_data, old_tasks = ((0,(1,0))) )

main(RegType.LFL, DsType.Emnist)