import os
import argparse
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
# from utils.funcs import *
from supp.Parser import *
from supp.get_dataset import *
from supp.create_model import *
from supp.FlagAt import *
from supp.logger import *
from supp.data_functions import *
from supp.loss_and_accuracy import *
from supp.visuialize_predctions import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Training_flag:
    def __init__(self,train_all_model, train_arg,task_embedding,head_learning):
        self.train_all_model = train_all_model
        self.train_arg = train_arg
        self.task_embedding = task_embedding
        self.head_learning = head_learning

def Get_learned_params(model, training_flag, embedding_idx):
    learned_params = []
    if training_flag.task_embedding:
        learned_params.extend(model.module.task_embedding[embedding_idx] )
    if training_flag.head_learning:
        learned_params.extend(model.module.transfer_learning[embedding_idx])
    if training_flag.train_arg:
        learned_params.extend(model.module.tdmodel.argument_embedding[embedding_idx])
    if training_flag.train_all_model:
        learned_params = list(model.parameters())
    return learned_params

def train_omniglot(parser, embedding_idx, the_datasets, training_flag):
    set_datasets_measurements(the_datasets, Measurements, parser, parser.model)
    cudnn.benchmark = True  # TODO:understand what it is.
    # Deciding which parameters will be trained: if True all the model otherwise,only the task embedding.
    learned_params = Get_learned_params(parser.model, training_flag,embedding_idx)
    # Training the learned params of the model.
    create_optimizer_and_sched(parser,learned_params)

    train_model(parser, the_datasets, learned_params, embedding_idx)

def train_cifar10(parser, embedding_idx, the_datasets):
    set_datasets_measurements(the_datasets, Measurements, parser, parser.model)
    cudnn.benchmark = True  # TODO:understand what it is.
    # Deciding which parameters will be trained: if True all the model otherwise,only the task embedding.
    learned_params = parser.model.parameters()
    # Training the learned params of the model.
    return train_model(parser, the_datasets, learned_params, embedding_idx)

def main_cifar10(lr = 0.01, wd = 0.0, lr_decay = 1.0,language_idx = 0):
    parser = GetParser(DsType.Cifar10,lr , wd, lr_decay, language_idx,use_bu1_loss = False)
    print_detail(parser)                       
    data_path = '/home/sverkip/data/BU-TD/yonathan/training_cifar10/data/processed'
    # Create the data for right.
    [the_datasets, _, test_dl, _, _] = get_dataset_cifar(parser,data_path)
    return train_cifar10(parser,embedding_idx = 0,the_datasets = the_datasets)

def main_omniglot(language_idx,train_right,train_left):
    parser = GetParser(ds_type = DsType.Omniglot,language_idx = language_idx, use_bu1_loss = False)
    print_detail(parser)
    embedding_idx = 0
    data_path = '/home/sverkip/data/BU-TD/yonathan/omniglot/data/samples/6_extended_' + str(language_idx)
    # Create the data for right.
    root = '/home/sverkip/data/BU-TD/yonathan/training_cifar10/data/processed'
    [the_datasets, _, test_dl, _, _] = get_dataset_for_spatial_realtions(parser, data_path, embedding_idx, 0)
    path_loading = '5R/model24.pt'
    model_path = parser.results_dir
    load_model(parser, model_path, path_loading, load_optimizer_and_schedular=False);
    # Training Right.
    if train_right:
        parser.EPOCHS = 30
        training_flag = Training_flag(train_all_model=False,train_arg = True, task_embedding = False, head_learning = True)
        train_omniglot(parser,embedding_idx = 0,the_datasets = the_datasets,training_flag = training_flag)
     #   acc = accuracy(parser, test_dl)
      #  print("Done training right, with accuracy : " + str(acc))
    # Training Left.
    if train_left:
        parser.EPOCHS = 80
        [the_datasets, _, test_dl, _, _ ] = get_dataset_for_spatial_realtions(parser, data_path, embedding_idx,1)
        training_flag = Training_flag(train_all_model = False, train_arg = False, task_embedding = True, head_learning = True)
        train_omniglot(parser, embedding_idx = 0, the_datasets = the_datasets, training_flag=training_flag)
      #  acc = accuracy(parser, test_dl)
    #print("Done training left, with accuracy : " + str(acc))

def main_emnist(language_idx,train_right,train_left):
    parser = GetParser(ds_type = DsType.Emnist,language_idx=language_idx,use_bu1_loss = True)
    print_detail(parser)
    embedding_idx = 0
    data_path = '/home/sverkip/data/BU-TD/yonathan/emnist/data/samples/6_extended_test_' + str(language_idx)
    # Create the data for right.
    [the_datasets, _, _,test_dl,_, _, _] = get_dataset_for_spatial_realtions(parser, data_path, 0, 0)
    # Training Right.
    if train_right:

        parser.EPOCHS = 25
        training_flag = Training_flag(train_all_model=True, train_arg=True, task_embedding=True, head_learning=True)
        train_omniglot(parser, embedding_idx=0, the_datasets=the_datasets, training_flag=training_flag)
      #  acc = accuracy(parser, test_dl)
       # print("Done training right, with accuracy : " + str(acc))
    # Training Left.
    if train_left:
        parser.EPOCHS = 80
        [the_datasets, _, _, test_dl,_, _, _] = get_dataset_for_spatial_realtions(parser, data_path, 0 , 1)
        training_flag = Training_flag(train_all_model=False, train_arg=False, task_embedding=True, head_learning=True)
        train_omniglot(parser, embedding_idx=0, the_datasets=the_datasets, training_flag=training_flag)
       # acc = accuracy(parser, test_dl)
   # print("Done training left, with accuracy : " + str(acc))

# TODO - CHNAGE THE RESULTS DIR INTO 4 DIRS.
#main_omniglot(17,True,True)
main_emnist(0,True,True)
#main_cifar10()