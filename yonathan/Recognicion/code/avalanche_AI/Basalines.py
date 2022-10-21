import os
<<<<<<< HEAD
# TODO  - GET RID OF THIS.
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(r'/home/sverkip/data/BU-TD/yonathan/Recognicion/code/')
from avalanche.benchmarks.generators.benchmark_generators import nc_benchmark
from avalanche.evaluation.metrics import loss_metrics
from avalanche.logging import WandBLogger, InteractiveLogger, TextLogger
from avalanche.training.plugins import LRSchedulerPlugin
from avalanche.training.plugins.evaluation import EvaluationPlugin
from supp.pytorch_lightning_model_and_checkpoints import CheckpointSaver
=======
from datetime import datetime
from pathlib import Path

from avalanche.benchmarks.generators.benchmark_generators import nc_benchmark
from avalanche.evaluation.metrics import loss_metrics
from avalanche.logging import WandBLogger
from avalanche.training.plugins import LRSchedulerPlugin
from avalanche.training.plugins.evaluation import EvaluationPlugin

from avalanche_AI.supp_avalanche_AI.checkpoints import CheckpointSaver
>>>>>>> 315b11ac3016dc72662fd8ca96881ae68c5cda6d
from classes import RegType
from supp.Parser import GetParser
from supp.general_functions import create_optimizer_and_sched, Get_learned_params
from supp.get_dataset import get_dataset_for_spatial_realtions
from supp.loss_and_accuracy import accuracy
from supp.models import ResNet
<<<<<<< HEAD
from supp.pytorch_lightning_model_and_checkpoints import load_model
from supp_avalanche_AI.Plugins.Accuracy_plugin import accuracy_metrics
from supp_avalanche_AI.strategy_wrappers import MyEWC as EWC, MySI as SI, MyLFL as LFL, Mylwf as LWF, MyMAS as MAS, MyRWALK as RWALK
=======
from supp.training_functions import load_model
from supp_avalanche_AI.Plugins.Accuracy_plugin import accuracy_metrics
from supp_avalanche_AI.strategy_wrappers import MyEWC as EWC, MySI as SI, MyLFL as LFL, Mylwf as LWF, MyMAS as MAS
>>>>>>> 315b11ac3016dc72662fd8ca96881ae68c5cda6d

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

task = 2

<<<<<<< HEAD
reg_type = RegType.SI

parser = GetParser(task_idx = 0, direction_idx = 'up', model_type = ResNet, Begin_with_pretrained_model = True, use_lateral_bu_td=False, use_lateral_td_bu=False)
=======
reg_type = RegType.EWC

parser = GetParser( language_idx= 0, direction = 'up', model_type = ResNet, Begin_with_pretrained_model = True)
>>>>>>> 315b11ac3016dc72662fd8ca96881ae68c5cda6d
project_path = Path(__file__).parents[2]
project_data_path = os.path.join(project_path,'data/{}'.format('emnist'))
data_path =  os.path.join(project_data_path, 'samples/18_extended')
##
<<<<<<< HEAD
[ train_dl ,  test_dl, val_dl , train_dataset, test_dataset, val_dataset] = get_dataset_for_spatial_realtions(parser, data_path, lang_idx = 0, direction = 0)

parser.old_dataset = train_dataset
##
[ train_dl ,  test_dl, val_dl , train_dataset, test_dataset, val_dataset] = get_dataset_for_spatial_realtions(parser, data_path, lang_idx = 0, direction = task)
=======
[the_datasets, train_dl ,  test_dl, val_dl , train_dataset, test_dataset, val_dataset] = get_dataset_for_spatial_realtions(parser, data_path, lang_idx = 0, direction = 0)

parser.old_dataset = train_dataset
##
[the_datasets, train_dl ,  test_dl, val_dl , train_dataset, test_dataset, val_dataset] = get_dataset_for_spatial_realtions(parser, data_path, lang_idx = 0, direction = task)
>>>>>>> 315b11ac3016dc72662fd8ca96881ae68c5cda6d
#
scenario = nc_benchmark(train_dataset = [train_dataset], test_dataset = [test_dataset], n_experiences = 1, shuffle = True, task_labels = False,one_dataset_per_exp = True)
learned_params = Get_learned_params(parser.model, task_id = task)

parser.optimizer, parser.scheduler = create_optimizer_and_sched(parser, parser.model.parameters())
# Loading a pretrained model.
<<<<<<< HEAD
path_loading = 'model2/ResNet_epoch19_direction=2.pt'

model_path = os.path.join(project_data_path,  'LWF')

#load_model(parser.model, model_path, path_loading, load_optimizer_and_schedular=False);

#print(accuracy(parser, test_dl))
#TL = TextLogger(open('/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/results/log.txt','a'))
IC = InteractiveLogger()
loggers = [WandBLogger(project_name="avalanche_EWC", run_name="train")]
=======
path_loading = 'ResNet_right/model_right_best.pt'

model_path = os.path.join(project_data_path,  'results')

load_model(parser.model, model_path, path_loading, load_optimizer_and_schedular=False);

print(accuracy(parser, test_dl))

loggers = [WandBLogger(project_name="avalanche_LFL", run_name="train")]
>>>>>>> 315b11ac3016dc72662fd8ca96881ae68c5cda6d

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
<<<<<<< HEAD
    strategy = SI(parser, checkpoint = checkpoint, test_dl =test_dl, device='cuda', evaluator=evaluator, train_epochs=1,  plugins=[schedu], eval_every=-1)
=======
    strategy = SI(parser,  train_mb_size=10, device='cuda', evaluator=evaluator, train_epochs=1,  plugins=[schedu], eval_every=-1)
>>>>>>> 315b11ac3016dc72662fd8ca96881ae68c5cda6d

if reg_type is RegType.LWF:
    strategy = LWF(parser,checkpoint = checkpoint, test_dl = test_dl, device='cuda', evaluator=evaluator, train_epochs=1,  plugins=[schedu], eval_every=-1)

if reg_type is RegType.LFL:
    strategy = LFL(parser,checkpoint = checkpoint, test_dl=test_dl, device='cuda', evaluator=evaluator, train_epochs=1,  plugins=[schedu], eval_every=-1)

if reg_type is RegType.MAS:
    strategy = MAS(parser,checkpoint = checkpoint, test_dl=test_dl, device='cuda', evaluator=evaluator, train_epochs=1,  plugins=[schedu], eval_every=-1)

<<<<<<< HEAD
if reg_type is RegType.RWALK:
    strategy = RWALK(parser,checkpoint = checkpoint, test_dl=test_dl, device='cuda', evaluator=evaluator, train_epochs=1,  plugins=[schedu], eval_every=-1)
=======
>>>>>>> 315b11ac3016dc72662fd8ca96881ae68c5cda6d
epochs = parser.epochs

for experience in scenario.train_stream:
 for epoch in range(epochs):
     strategy.train(experience)
     print("Done, one experimnet")
     strategy.eval(scenario.test_stream)
