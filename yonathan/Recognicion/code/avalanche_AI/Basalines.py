from avalanche.training.plugins.evaluation import EvaluationPlugin
from supp_avalanche_AI.SupervisedTemplates import MyEWC as EWC, MySI as SI, MyLFL as LFL, Mylwf as LWF
from avalanche.benchmarks.generators.benchmark_generators import nc_benchmark
from supp.general_functions import create_optimizer_and_sched
from supp.Parser import GetParser
from supp.get_dataset import get_dataset_for_spatial_realtions
from supp.FlagAt import Flag, DsType, Model_Options_By_Flag_And_DsType
from avalanche.training.plugins import LRSchedulerPlugin
from avalanche.logging import WandBLogger
from parser import Get_basilnes_parser
from supp.models import ResNet
from supp.training_functions import load_model
import os
import numpy as np
from avalanche.logging import TensorboardLogger,TextLogger,InteractiveLogger
from supp_avalanche_AI.Accuracy_plugin import accuracy_metrics
from avalanche.evaluation.metrics import loss_metrics
from classes import RegType
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
#from pytorch_lightning.loggers import WandbLogger
###########################
reg_type = RegType.EWC
opts = Model_Options_By_Flag_And_DsType(Flag=Flag.NOFLAG, DsType=DsType.Emnist)
parser = GetParser(opts=opts, language_idx= 0,direction = 'left',model_type=ResNet)
data_path = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/samples/18_extended'
# Create the data for right.
tmpdir = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/results/'

[the_datasets, train_dl ,  test_dl, val_dl , train_dataset, test_dataset, val_dataset] = get_dataset_for_spatial_realtions(parser, data_path,lang_idx = 0, direction = 2)
#
opts.optimizer, opts.scheduler = create_optimizer_and_sched(parser, parser.model.parameters())
#
train_dataset.targets = [0 for _ in range(len(train_dataset))] # [train_dataset[i][1] for i in range(len(train_dataset))]
test_dataset.targets = [0 for _ in range(len(test_dataset))] # [test_dataset[i][1] for i in range(len(test_dataset))]
val_dataset.targets = [0 for _ in range(len(val_dataset))] # [val_dataset[i][1] for i in range(len(val_dataset))]
###############################
# Loading a pretrained model.
path_loading = 'Model0_right_test_stronger_emb10.10.2022 12:09:18/model_right_best.pt'
model_path = parser.results_dir
load_model(parser.model, model_path, path_loading, load_optimizer_and_schedular=False);
###############################
scenario = nc_benchmark(train_dataset = [train_dataset], test_dataset = [test_dataset], n_experiences = 1, shuffle = True, task_labels = False,one_dataset_per_exp = True)
import torch
model = parser.model
optimizer = opts.optimizer
schedu = LRSchedulerPlugin(opts.scheduler)
criterion = parser.loss_fun

loggers = []

loggers.append(TensorboardLogger())

# log to text file
loggers.append(TextLogger(open('log.txt', 'a')))

# print to stdout
loggers.append(InteractiveLogger())
Baseline_parser = Get_basilnes_parser()
loggers = []
loggers.append(WandBLogger(project_name="avalanche", run_name="train"))

#logger = WandbLogger(project = "My_first_project_5.10", job_type = 'train',save_dir='/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/results/')
#############
class CheckpointSaver:
    def __init__(self, dirpath, decreasing=False, top_n=5):
        """
        dirpath: Directory path where to store all model weights
        decreasing: If decreasing is `True`, then lower metric is better
        top_n: Total number of models to track based on validation metric value
        """
        if not os.path.exists(dirpath): os.makedirs(dirpath)
        self.dirpath = dirpath
        self.top_n = top_n
        self.decreasing = decreasing
        self.top_model_paths = []
        self.best_metric_val = np.Inf if decreasing else -np.Inf

    def __call__(self, model, epoch, metric_val):
        model_path = os.path.join(self.dirpath, model.__class__.__name__ + f'_epoch{epoch}.pt')
        save = metric_val<self.best_metric_val if self.decreasing else metric_val>self.best_metric_val
        if save:
           # logging.info(f"Current metric value better than {metric_val} better than best {self.best_metric_val}, saving model at {model_path}")
            self.best_metric_val = metric_val
            torch.save(model.state_dict(), model_path)
            self.top_model_paths.append({'path': model_path, 'score': metric_val})
            self.top_model_paths = sorted(self.top_model_paths, key=lambda o: o['score'], reverse=not self.decreasing)
        if len(self.top_model_paths) > self.top_n:
            self.cleanup()

    def cleanup(self):
        to_remove = self.top_model_paths[self.top_n:]
        print("Removing extra models")
        for o in to_remove:
            os.remove(o['path'])
        self.top_model_paths = self.top_model_paths[:self.top_n]
from datetime import datetime
now = datetime.now()
dt_string = now.strftime("%d.%m.%Y %H:%M:%S")
checkpoint = CheckpointSaver('/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/results/First_ewc_model'+dt_string)

#############
Baseline_parser = Get_basilnes_parser()
evaluator = EvaluationPlugin( accuracy_metrics('MACW',minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),benchmark = scenario, loggers=loggers )
Ignored_params = list(model.taskhead.named_parameters())

if reg_type is RegType.EWC:
    strategy =  EWC(model, optimizer, criterion,checkpoint = checkpoint, test_dl=test_dl, start_from_regulization = True, Ignored_params = Ignored_params,  ewc_lambda = Baseline_parser.ewc_lambda, train_mb_size = 10,   device = 'cuda',evaluator = evaluator, train_epochs=1, plugins=[schedu], eval_every=-1)

if reg_type is RegType.SI:
    strategy = SI(model, optimizer, criterion, si_lambda = Baseline_parser.si_lambda, train_mb_size=10, device='cuda', evaluator=evaluator, train_epochs=1,  plugins=[schedu], eval_every=1)

if reg_type is RegType.LWF:
    strategy = LWF(model, optimizer, criterion, start_from_regulization=True, Ignored_params=Ignored_params, ewc_lambda=1e18, train_mb_size=10, device='cuda', evaluator=evaluator, train_epochs=10,  plugins=[schedu], eval_every=-1)

if reg_type is RegType.LFL:
    strategy = LFL(model, optimizer, criterion, start_from_regulization=True, Ignored_params=Ignored_params, ewc_lambda=1e18, train_mb_size=10, device='cuda', evaluator=evaluator, train_epochs=10,  plugins=[schedu], eval_every=1)
epochs = 20
for epoch in range(epochs):
    for experience in scenario.train_stream:
     strategy.train(experience)
     print("Done, one experimnet")
     strategy.eval(scenario.test_stream)
