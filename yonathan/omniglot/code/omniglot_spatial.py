import os
import argparse
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
# from utils.funcs import *
from supp.Parser import *
from supp.get_dataset import *
from supp.create_model import *
from supp.FlagAt import *
from supp.omniglot_dataset import inputs_to_struct as inputs_to_struct
from supp.logger import *
from supp.data_functions import *
from supp.loss_and_accuracy import *
from supp.visuialize_predctions import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def Get_learned_params(model, training_flag, embedding_idx):
    learned_params = []
    if training_flag.task_embedding:
        learned_params.extend(model.module.task_embedding[embedding_idx] )
    if training_flag.head_learning:
        learned_params.extend(model.module.transfer_learning[embedding_idx])
    if training_flag.train_arg:
        learned_params.extend(model.module.tdmodel.argument_embedding[embedding_idx])
    if training_flag.train_all_model:
        learned_params = model.parameters()
    return learned_params

def train_omniglot(parser, embedding_idx, the_datasets, training_flag):
    set_datasets_measurements(the_datasets, Measurements, parser, parser.model)
    cudnn.benchmark = True  # TODO:understand what it is.
    # Deciding which parameters will be trained: if True all the model otherwise,only the task embedding.
    learned_params = Get_learned_params(parser.model, training_flag,embedding_idx)
    # Training the learned params of the model.
    create_optimizer_and_sched(parser,learned_params)

    train_model(parser, the_datasets, learned_params, embedding_idx)

class Training_flag:
    def __init__(self,train_all_model, train_arg,task_embedding,head_learning):
        self.train_all_model = train_all_model
        self.train_arg = train_arg
        self.task_embedding = task_embedding
        self.head_learning = head_learning

def main(language_idx,train_right,train_left):
    parser = GetParser(language_idx)

    print_detail(parser)

    embedding_idx = 0
    data_path = '/home/sverkip/data/BU-TD/yonathan/data/samples/new_samples/6_extended_' + str(language_idx)

    # Create the data for right.
    [the_datasets, _, test_dl, _, _] = get_dataset(embedding_idx, 0 , parser, data_fname=data_path)
    path_loading = '5R/model24.pt'
    model_path = parser.results_dir
    load_model(parser, model_path, path_loading, load_optimizer_and_schedular=False);
    # Training Right.

    if train_right:
        parser.EPOCHS = 15
        training_flag = Training_flag(train_all_model=False,train_arg = True, task_embedding = False, head_learning = True)
        train_omniglot(parser,embedding_idx = 0,the_datasets = the_datasets,training_flag = training_flag)
        acc = accuracy(parser, test_dl)
        print("Done training right, with accuracy : " + str(acc))
    # Training Left.
    if train_left:
        parser.EPOCHS = 80
        [the_datasets, _, test_dl, _, _ ] = get_dataset(embedding_idx, 1, parser, data_fname=data_path)
        training_flag = Training_flag(train_all_model = False, train_arg = False, task_embedding = True, head_learning = True)
        train_omniglot(parser, embedding_idx = 0, the_datasets = the_datasets, training_flag=training_flag)
        acc = accuracy(parser, test_dl)
    print("Done training left, with accuracy : " + str(acc))

main(17,True,True)

# 6_extended_[27, 5, 42, 18].
# Temp place
##############################################
# print(accuracy(parser, test_dl))
# visualize(parser, train_dataset)
# /home/sverkip/data/Omniglot/data/new_samples/T
# print(num_params(learned_params))
# print(accuracy_one_language(args.model, args.test_dl))
# /home/sverkip/data/Omniglot/data/results/DS=Four_languages_embedding_idx=029.06.2022 11:55:49/model5.pt
# 'DS=8_extended_exp[27, 5, 42]_embedding_idx=029.06.2022 15:21:58'
###############################################
