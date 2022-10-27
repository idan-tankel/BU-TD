import os
import sys
from datetime import datetime
from pathlib import Path
# TODO  - GET RID OF THIS.
sys.path.append(r'/home/sverkip/data/BU-TD/yonathan/Recognicion/code/')
from avalanche.benchmarks.generators.benchmark_generators import nc_benchmark
from avalanche.evaluation.metrics import loss_metrics
from avalanche.logging import WandBLogger, InteractiveLogger
from avalanche.training.plugins import LRSchedulerPlugin
from avalanche.training.plugins.evaluation import EvaluationPlugin
from supp.pytorch_lightning_model_and_checkpoints import CheckpointSaver, load_model
from classes import RegType
from supp.Parser import GetParser
from supp.utils import create_optimizer_and_scheduler, Get_learned_params
from supp.get_dataset import get_dataset_for_spatial_realtions
from supp.models import ResNet
from avalanche_AI.training.Plugins.Evaluation import accuracy_metrics
from avalanche_AI.training.supervised.strategy_wrappers import MyEWC as EWC, MySI as SI, MyLFL as LFL, Mylwf as LWF, MyMAS as MAS, MyRWALK as RWALK,AGEM, GEM,MyIL
from supp.loss_and_accuracy import accuracy
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

reg_type = RegType.RWALK

parser = GetParser(task_idx = 0, direction_idx = 'up', model_type = ResNet, Begin_with_pretrained_model = True, use_lateral_bu_td=False, use_lateral_td_bu=False)
project_path = Path(__file__).parents[2]
project_data_path = os.path.join(project_path,'data/{}'.format('emnist'))
data_path =  os.path.join(project_data_path, 'samples/18_extended')
#
[ t_ ,  test_dl, _ , train_dataset_right, test_dataset_right, _ ] = get_dataset_for_spatial_realtions(parser, data_path, lang_idx = 0, direction = 0)

parser.old_dataset = train_dataset_right
#
#print(parser.old_dataset)

task = 2

[ _ ,  test_dl, _ , train_dataset_up, test_dataset_up, _ ] = get_dataset_for_spatial_realtions(parser, data_path, lang_idx = 0, direction = task)
#
if reg_type is RegType.SI or reg_type is RegType.RWALK:
    scenario = nc_benchmark(train_dataset = [train_dataset_right, train_dataset_up], test_dataset = [test_dataset_right, test_dataset_up], n_experiences = 2, shuffle = True, task_labels = False,one_dataset_per_exp = True)
else:
    scenario = nc_benchmark(train_dataset=[ train_dataset_up],    test_dataset=[test_dataset_up], n_experiences = 1, shuffle=True,  task_labels=False, one_dataset_per_exp=True)

learned_params = Get_learned_params(parser.model, task_id = task) # Train only the desired params.

parser.optimizer, parser.scheduler = create_optimizer_and_scheduler(parser, learned_params)
# Loading a pretrained model.
path_loading = 'results/Model_all_right_better/ResNet_epoch20_direction=0.pt'

#path_loading = 'MAS_model25.10.2022 15:54:01/ResNet_epoch18_direction=2.pt'

load_model(parser.model, project_data_path, path_loading, load_optimizer_and_schedular=False);

#parser.model = MyMultiTaskModule(parser.model)

#print(accuracy(parser, test_dl))

#TL = TextLogger(open('/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/results/log.txt','a'))

IC = InteractiveLogger()
loggers = [WandBLogger(project_name="avalanche_EWC", run_name="train",dir='/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/results',path = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/results/checkpoint' ), IC]

now = datetime.now()
dt_string = now.strftime("%d.%m.%Y %H:%M:%S")
Baseline_folder = os.path.join(project_data_path, reg_type.Enum_to_name())
Model_folder = os.path.join(Baseline_folder, "lambda=" + str(reg_type.class_to_reg(parser)))
checkpoint = CheckpointSaver(Model_folder)
print("Code script saved")
schedu = LRSchedulerPlugin(parser.scheduler)

evaluator = EvaluationPlugin(accuracy_metrics(parser,minibatch=True, epoch=True, experience=True, stream=True),loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),benchmark = scenario,loggers=loggers )

if reg_type is RegType.EWC:
    strategy =  EWC(parser, checkpoint = checkpoint, test_dl=test_dl, device = 'cuda',evaluator = evaluator, train_epochs=1, plugins=[schedu], eval_every=-1)

if reg_type is RegType.SI:
    strategy = SI(parser, checkpoint = checkpoint, test_dl =test_dl, device='cuda', evaluator=evaluator, train_epochs=1,  plugins=[schedu], eval_every=-1)

if reg_type is RegType.LWF:
    strategy = LWF(parser,checkpoint = checkpoint, test_dl = test_dl, device='cuda', evaluator=evaluator, train_epochs=1,  plugins=[schedu], eval_every=-1)

if reg_type is RegType.LFL:
    strategy = LFL(parser,checkpoint = checkpoint, test_dl=test_dl, device='cuda', evaluator=evaluator, train_epochs=1,  plugins=[schedu], eval_every=-1)

if reg_type is RegType.MAS:
    strategy = MAS(parser,checkpoint = checkpoint, test_dl=test_dl, device='cuda', evaluator=evaluator, train_epochs=1,  plugins=[schedu], eval_every=-1)

if reg_type is RegType.RWALK:
    strategy = RWALK(parser,checkpoint = checkpoint, test_dl=test_dl, device='cuda', evaluator=evaluator, train_epochs=1,  plugins=[schedu], eval_every=-1)

if reg_type is RegType.AGEM:
    strategy = AGEM(parser = parser, old_data = parser.old_dataset, checkpoint=checkpoint, test_dl=test_dl, device='cuda', evaluator=evaluator, train_epochs=1,
                     plugins=[schedu], eval_every=-1)

if reg_type is RegType.GEM:
    strategy = GEM(parser = parser, old_data = parser.old_dataset, checkpoint=checkpoint, test_dl=test_dl, device='cuda', evaluator=evaluator, train_epochs=1, plugins=[schedu], eval_every=-1)

if reg_type is RegType.IL:
    strategy = MyIL(parser = parser, old_data = parser.old_dataset, checkpoint=checkpoint, test_dl=test_dl, device='cuda', evaluator=evaluator, train_epochs=1, plugins=[schedu], eval_every=-1)

epochs = parser.epochs
if reg_type is RegType.SI or reg_type is RegType.RWALK:
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
     for epoch in range(epochs):
         strategy.train(exp_train, [exp_test])
         print("Done, one experimnet")
         strategy.eval(exp_test)
