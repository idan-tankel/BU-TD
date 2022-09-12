import os
import argparse
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
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

    def Get_learned_params(self,model, embedding_idx):
        learned_params = []
        if self.task_embedding:
            learned_params.extend(model.module.task_embedding[embedding_idx])
        if self.head_learning:
            learned_params.extend(model.module.transfer_learning[embedding_idx])
        if self.train_arg:
            learned_params.extend(model.module.tdmodel.argument_embedding[embedding_idx])
        if self.train_all_model:
            learned_params = list(model.parameters())
        return learned_params


def train_omniglot(parser, embedding_idx, the_datasets, training_flag):
    set_datasets_measurements(the_datasets, Measurements, parser, parser.model)
    cudnn.benchmark = True  # TODO:understand what it is.
    # Deciding which parameters will be trained: if True all the model otherwise,only the task embedding.
    learned_params =training_flag.Get_learned_params(parser.model, embedding_idx)
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
    opts = Model_Options_By_Flag_And_DsType(Flag=Flag.SF, DsType=DsType.Omniglot)
    parser = GetParser(opts=opts, language_idx=0)
    print_detail(parser)
    embedding_idx = 0
    data_path = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/omniglot/samples/6_extended_testing_0'
    # Create the data for right.
    [the_datasets, _,  test_dl, _ , _ , _, _] = get_dataset_for_spatial_realtions(parser, data_path, embedding_idx = 0, direction = 1)
    # Training Right.
    path_loading = 'Model012.09.2022 10:36:12/model14_right.pt'
    model_path = parser.results_dir
    load_model(parser, model_path, path_loading, load_optimizer_and_schedular=False);
    load_running_stats(parser.model, task_emb_id = 0);
  #  acc = accuracy(parser, test_dl)
  #  print("Done training right, with accuracy : " + str(acc))
    if train_right:
        parser.EPOCHS = 20
        training_flag = Training_flag(train_all_model=True, train_arg=True, task_embedding=False, head_learning=True)
        train_omniglot(parser, embedding_idx=0, the_datasets=the_datasets, training_flag=training_flag)

    if train_left:
        parser.EPOCHS = 100
        [the_datasets, _, _, _, _, _, _] = get_dataset_for_spatial_realtions(parser, data_path, embedding_idx = 0, direction = 1 )
        training_flag = Training_flag(train_all_model=False, train_arg=False, task_embedding=True, head_learning=True)
        train_omniglot(parser, embedding_idx = 0, the_datasets=the_datasets, training_flag=training_flag)

def main_emnist(language_idx,train_right,train_left):
    opts = Model_Options_By_Flag_And_DsType(Flag = Flag.SF, DsType = DsType.Emnist)
    parser = GetParser(opts = opts, language_idx = language_idx)
    print_detail(parser)

    data_path = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/samples/6_extended' + str(language_idx)
    # Create the data for right.
    [the_datasets, _, _,test_dl,_, _, _] = get_dataset_for_spatial_realtions(parser, data_path,embedding_idx =  0, direction =  0)
    # Training Right.
    if train_right:
        parser.EPOCHS = 20
        training_flag = Training_flag(train_all_model=True, train_arg=True, task_embedding=True, head_learning=True)
        train_omniglot(parser, embedding_idx = 0, the_datasets=the_datasets, training_flag=training_flag)

    if train_left:
        parser.EPOCHS = 100
        [the_datasets, _, _, _,_, _, _] = get_dataset_for_spatial_realtions(parser, data_path, embedding_idx =  1, direction =  1)
        training_flag = Training_flag(train_all_model=False, train_arg=False, task_embedding=True, head_learning=True)
        train_omniglot(parser, embedding_idx = 1, the_datasets=the_datasets, training_flag=training_flag)
    print("Done training left, with accuracy : " + str(acc))

def main_FashionEmnist(language_idx,train_right,train_left):
    opts = Model_Options_By_Flag_And_DsType(Flag = Flag.SF, DsType = DsType.FashionMnist)
    parser = GetParser(opts = opts, language_idx = language_idx)
    print_detail(parser)
    data_path = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/FashionMnist/samples/6_extended_testing_' + str(language_idx)
    # Create the data for right.
    [the_datasets, _, _,test_dl,_, _, _] = get_dataset_for_spatial_realtions(parser, data_path,embedding_idx =  0, direction =  0)
    if train_right:
        parser.EPOCHS = 20
        training_flag = Training_flag(train_all_model=True, train_arg=True, task_embedding=True, head_learning=True)
        train_omniglot(parser, embedding_idx = 0, the_datasets=the_datasets, training_flag=training_flag)

    if train_left:
        parser.EPOCHS = 100
        [the_datasets, _, _, _,_, _, _] = get_dataset_for_spatial_realtions(parser, data_path, embedding_idx =  1, direction =  1)
        training_flag = Training_flag(train_all_model=False, train_arg=False, task_embedding=True, head_learning=True)
        train_omniglot(parser, embedding_idx = 1, the_datasets=the_datasets, training_flag=training_flag)
    print("Done training left, with accuracy : " + str(acc))

# TODO - CHNAGE THE RESULTS DIR INTO 4 DIRS.
#main_FashionEmnist(0, True, True)
main_omniglot(24,True,True)
#main_emnist(0,True,True)
#main_cifar10()









'''
path_loading = 'Model_without_bias/model_latest_left.pt'
model_path = parser.results_dir
load_model(parser, model_path, path_loading, load_optimizer_and_schedular=False);
load_running_stats(parser.model, task_emb_id = 1);
acc = accuracy(parser, test_dl)
print("Done training right, with accuracy : " + str(acc))
'''