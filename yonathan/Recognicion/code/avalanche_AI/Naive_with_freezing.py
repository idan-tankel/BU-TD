import os

from avalanche.benchmarks.generators.benchmark_generators import nc_benchmark
from avalanche.evaluation.metrics import loss_metrics
from avalanche.logging import WandBLogger
from avalanche.training.plugins import LRSchedulerPlugin
from avalanche.training.plugins.evaluation import EvaluationPlugin
from supp.training_functions import load_model
from supp_avalanche_AI.Accuracy_plugin import accuracy_metrics

from supp.Dataset_and_model_type_specification import Flag, DsType, Model_Options_By_Flag_And_DsType
from supp.Parser import GetParser
from supp.general_functions import create_optimizer_and_sched
from supp.get_dataset import get_dataset_for_spatial_realtions
from supp.models import ResNet
from supp_avalanche_AI.strategy_wrappers import Naive_with_freezing

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch.optim as optim
###########################

opts = Model_Options_By_Flag_And_DsType(Flag=Flag.NOFLAG, DsType=DsType.Emnist)
parser = GetParser(opts=opts, language_idx= 0,direction = 'left',model_type=ResNet)
data_path = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/samples/6_extended'
# Create the data for right.
tmpdir = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/results/'

[the_datasets, train_dl ,  test_dl, val_dl , train_dataset, test_dataset, val_dataset] = get_dataset_for_spatial_realtions(parser, data_path,lang_idx = 0, direction = 1)
#
opts.optimizer, opts.scheduler = create_optimizer_and_sched(parser, parser.model.parameters())
#
train_dataset.targets = [0 for _ in range(len(train_dataset))] # [train_dataset[i][1] for i in range(len(train_dataset))]
test_dataset.targets = [0 for _ in range(len(test_dataset))] # [test_dataset[i][1] for i in range(len(test_dataset))]
val_dataset.targets = [0 for _ in range(len(val_dataset))] # [val_dataset[i][1] for i in range(len(val_dataset))]
###############################
# Loading a pretrained model.
path_loading = 'Model0_right_test_stronger_emb07.10.2022 15:29:41/model_right_best.pt'
model_path = parser.results_dir
load_model(parser.model, model_path, path_loading, load_optimizer_and_schedular=False);
###############################
scenario = nc_benchmark(train_dataset = [train_dataset, train_dataset], test_dataset = [test_dataset, test_dataset], n_experiences = 2, shuffle = True, task_labels = False,one_dataset_per_exp = True)

model = parser.model
optimizer = opts.optimizer
schedu = LRSchedulerPlugin(opts.scheduler)
criterion = parser.loss_fun
loggers = []
optimizer_new = optim.Adam(model.taskhead.parameters())
loggers.append(WandBLogger(project_name="avalanche", run_name="train",path='/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/results/MyFirstCheckpoint.ckpt'))
evaluator = EvaluationPlugin( accuracy_metrics('MACW',minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),loggers = loggers )
Ignored_params = list(model.taskhead.named_parameters())
strategy =  Naive_with_freezing(model, [optimizer,optimizer_new], criterion, train_mb_size = 10,   device = 'cuda',evaluator = evaluator, train_epochs = [1,10], plugins=[schedu], eval_every=1)

for experience in scenario.train_stream:
    strategy.train(experience)
    print("Done, one experimnet")