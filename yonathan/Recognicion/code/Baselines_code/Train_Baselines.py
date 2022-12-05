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
from training.Data.Checkpoints import CheckpointSaver, load_model
from classes import RegType
from training.Data.Parser import GetParser
from training.Data.Structs import Training_flag
from avalanche.benchmarks.generators import dataset_benchmark
from training.Modules.Create_Models import create_model
from training.Utils import create_optimizer_and_scheduler

from training.Data.Get_dataset import get_dataset_for_spatial_relations
from training.Modules.Models import ResNet, BUTDModel
from avalanche_AI.training.Plugins.Evaluation import accuracy_metrics
from avalanche_AI.training.supervised.strategy_wrappers import MyEWC as EWC, LFL, LWF, MyMAS as MAS, MyRWALK as RWALK,SI
from training.Metrics.Accuracy import accuracy


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

reg_type = RegType.SI

task = (0,1)

parser = GetParser(task_idx = 0, direction_idx = 'up', model_type = ResNet)
parser.ns = [0,3,3,3]
parser.use_laterals_bu_td = False
parser.use_laterals_td_bu = False
project_path = Path(__file__).parents[2]
project_data_path = os.path.join(project_path,'Data_Creation/{}'.format('Emnist'))
data_path =  os.path.join(project_data_path, 'samples/(4,4)_extended_format')
#
Right_dict = get_dataset_for_spatial_relations(parser, data_path, lang_idx = 0, direction_tuple= (1, 0))


#
Up_dict = get_dataset_for_spatial_relations(parser, data_path, lang_idx = 0, direction_tuple= task)
#print(parser.old_dataset)

train_dataset_right = Right_dict['train_ds']
train_dataset_up = Up_dict['train_ds']
#
test_dataset_right = Right_dict['test_ds']
test_dataset_up = Up_dict['test_ds']

parser.model = create_model(parser)

parser.model.trained_tasks.add((0,(1,0)))
#
if reg_type is RegType.RWALK:
    scenario = dataset_benchmark(train_dataset = [train_dataset_right, train_dataset_up], test_dataset = [test_dataset_right, test_dataset_up], n_experiences = 2, shuffle = True, task_labels = False,one_dataset_per_exp = True)
else:
    scenario = dataset_benchmark([ train_dataset_right],  [test_dataset_right])

scenario = dataset_benchmark([ train_dataset_up], [test_dataset_up])

parser.prev_data = scenario.train_stream[0].dataset

learned_params = parser.model.parameters() # Train only the desired params.

parser.pretrained_model = copy.deepcopy(parser.model)

parser.optimizer, parser.scheduler = create_optimizer_and_scheduler(parser, learned_params,nbatches_train=len(Up_dict['train_dl']))
# Loading a pretrained model.
path_loading = 'results/Model_all_right_better/ResNet_epoch20_direction=0.pt'

#path_loading = 'MAS_model25.10.2022 15:54:01/ResNet_epoch18_direction=2.pt'

#load_model(parser.model, project_data_path, path_loading, load_optimizer_and_schedular=False);

#parser.model = MyMultiTaskModule(parser.model)

#print(Accuracy(parser, test_dl))

#TL = TextLogger(open('/home/sverkip/Data_Creation/BU-TD/yonathan/Recognicion/Data_Creation/emnist/results/log.txt','a'))

IC = InteractiveLogger()
loggers = [WandBLogger(project_name="avalanche_EWC", run_name="train",dir='/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/results',path = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/results/checkpoint' ), IC]

now = datetime.now()
dt_string = now.strftime("%d.%m.%Y %H:%M:%S")
Baseline_folder = os.path.join(project_data_path, str(reg_type))
Model_folder = os.path.join(Baseline_folder, "lambda=" + str(reg_type.class_to_reg(parser)))
checkpoint = CheckpointSaver(Model_folder)
print("Code script saved")


evaluator = EvaluationPlugin(accuracy_metrics(parser,minibatch=True, epoch=True, experience=True, stream=True),loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),benchmark = scenario,loggers=loggers )

if reg_type is RegType.EWC:
    strategy =  EWC(parser, checkpoint = checkpoint, device = 'cuda',evaluator = evaluator, eval_every=1,prev_model=parser.model)

if reg_type is RegType.LWF:
    strategy = LWF(parser,checkpoint = checkpoint, evaluator=evaluator, eval_every=1,prev_model=parser.model)

if reg_type is RegType.LFL:
    strategy = LFL(parser,checkpoint = checkpoint, evaluator=evaluator, eval_every=1,prev_model=parser.model)

if reg_type is RegType.MAS:
    strategy = MAS(parser,checkpoint = checkpoint, device='cuda', evaluator=evaluator, eval_every=1,prev_model=parser.model)

if reg_type is RegType.RWALK:
    strategy = RWALK(parser,checkpoint = checkpoint, test_dl=test_dl, device='cuda', evaluator=evaluator, train_epochs=1,  plugins=[schedu], eval_every=-1,prev_model=parser.model)

if reg_type is RegType.SI:
    head_params = [key for (key,_) in parser.model.named_parameters() if 'head' in key]
    strategy = SI(parser,checkpoint = checkpoint, device='cuda', evaluator=evaluator, eval_every=1,exc= head_params)

#epochs = parser.epochs
if reg_type is RegType.RWALK:
    for idx, (exp_train, exp_test) in enumerate(zip(scenario.train_stream, scenario.test_stream)):
     strategy.optimizer.zero_grad(set_to_none=True)  # Make the taskhead of previous task static.
     if idx == 0:
         strategy.EpochClock.max_epochs = 1
         strategy.EpochClock.train_exp_counter = 0
         strategy.train(exp_train)
         print("Done, one experimnet")
       #  strategy.eval(exp_test)
     else:
         strategy.EpochClock.train_exp_counter = 1
         strategy.EpochClock.max_epochs = 20
         strategy.EpochClock.just_initialized = True
         print("Switch to Up")
         for epoch in range(epochs):
             strategy.train(exp_train)
             print("Done, one experimnet")
             strategy.eval(exp_test)
else:
    for idx, (exp_train, exp_test) in enumerate(zip(scenario.train_stream, scenario.test_stream)):
     strategy.optimizer.zero_grad(set_to_none=True) # Make the taskhead of previous task static.
    # for epoch in range(epochs):
     strategy.train(exp_train,[exp_test])
     print("Done, one experiment")
     strategy.eval(exp_test)
     eval = strategy.evaluator
