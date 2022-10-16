import os
from datetime import datetime
from pathlib import Path

from avalanche.benchmarks.generators.benchmark_generators import nc_benchmark
from avalanche.evaluation.metrics import loss_metrics
from avalanche.logging import WandBLogger
from avalanche.training.plugins import LRSchedulerPlugin
from avalanche.training.plugins.evaluation import EvaluationPlugin

from avalanche_AI.supp_avalanche_AI.checkpoints import CheckpointSaver
from classes import RegType
from supp.Parser import GetParser
from supp.general_functions import create_optimizer_and_sched, Get_learned_params
from supp.get_dataset import get_dataset_for_spatial_realtions
from supp.loss_and_accuracy import accuracy
from supp.models import ResNet
from supp.training_functions import load_model
from supp_avalanche_AI.Plugins.Accuracy_plugin import accuracy_metrics
from supp_avalanche_AI.strategy_wrappers import MyEWC as EWC, MySI as SI, MyLFL as LFL, Mylwf as LWF, MyMAS as MAS

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

task = 2

reg_type = RegType.EWC

parser = GetParser( language_idx= 0, direction = 'up', model_type = ResNet, Begin_with_pretrained_model = True)
project_path = Path(__file__).parents[2]
project_data_path = os.path.join(project_path,'data/{}'.format('emnist'))
data_path =  os.path.join(project_data_path, 'samples/18_extended')
##
[the_datasets, train_dl ,  test_dl, val_dl , train_dataset, test_dataset, val_dataset] = get_dataset_for_spatial_realtions(parser, data_path, lang_idx = 0, direction = 0)

parser.old_dataset = train_dataset
##
[the_datasets, train_dl ,  test_dl, val_dl , train_dataset, test_dataset, val_dataset] = get_dataset_for_spatial_realtions(parser, data_path, lang_idx = 0, direction = task)
#
scenario = nc_benchmark(train_dataset = [train_dataset], test_dataset = [test_dataset], n_experiences = 1, shuffle = True, task_labels = False,one_dataset_per_exp = True)
learned_params = Get_learned_params(parser.model, task_id = task)

parser.optimizer, parser.scheduler = create_optimizer_and_sched(parser, parser.model.parameters())
# Loading a pretrained model.
path_loading = 'ResNet_right/model_right_best.pt'

model_path = os.path.join(project_data_path,  'results')

load_model(parser.model, model_path, path_loading, load_optimizer_and_schedular=False);

print(accuracy(parser, test_dl))

loggers = [WandBLogger(project_name="avalanche_LFL", run_name="train")]

now = datetime.now()
dt_string = now.strftime("%d.%m.%Y %H:%M:%S")
Baseline_folder = os.path.join(project_data_path, reg_type.Enum_to_name())
Model_folder = os.path.join(Baseline_folder, reg_type.Enum_to_name() + "_model")+dt_string
checkpoint = CheckpointSaver(Model_folder)

schedu = LRSchedulerPlugin(parser.scheduler)

evaluator = EvaluationPlugin(accuracy_metrics('MACW',minibatch=True, epoch=True, experience=True, stream=True),
 loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
 benchmark = scenario,
  loggers=loggers )

if reg_type is RegType.EWC:
    strategy =  EWC(parser, checkpoint = checkpoint, test_dl=test_dl, device = 'cuda',evaluator = evaluator, train_epochs=1, plugins=[schedu], eval_every=-1)

if reg_type is RegType.SI:
    strategy = SI(parser,  train_mb_size=10, device='cuda', evaluator=evaluator, train_epochs=1,  plugins=[schedu], eval_every=-1)

if reg_type is RegType.LWF:
    strategy = LWF(parser,checkpoint = checkpoint, test_dl = test_dl, device='cuda', evaluator=evaluator, train_epochs=1,  plugins=[schedu], eval_every=-1)

if reg_type is RegType.LFL:
    strategy = LFL(parser,checkpoint = checkpoint, test_dl=test_dl, device='cuda', evaluator=evaluator, train_epochs=1,  plugins=[schedu], eval_every=-1)

if reg_type is RegType.MAS:
    strategy = MAS(parser,checkpoint = checkpoint, test_dl=test_dl, device='cuda', evaluator=evaluator, train_epochs=1,  plugins=[schedu], eval_every=-1)

epochs = parser.epochs

for experience in scenario.train_stream:
 for epoch in range(epochs):
     strategy.train(experience)
     print("Done, one experimnet")
     strategy.eval(scenario.test_stream)
