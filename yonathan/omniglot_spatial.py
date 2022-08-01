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


def train_omniglot(embedding_idx, flag_at, processed_data, raw_data,model_name= None, path_loading=None, train_all_model=True,train_arg=False,transfer_learning=False,task_embedding=False,load_model_if_exists = False):
    # Getting the options for creating the model and the hyper-parameters.
    results_dir = '/home/sverkip/data/omniglot/data/results'
    parser = GetParser(flag_at, raw_data, processed_data, embedding_idx, results_dir,train_arg,1,model_name,load_model_if_exists)
    # Getting the dataset for the training.
    data_path = os.path.join('/home/sverkip/data/omniglot/data/new_samples', processed_data)
    [the_datasets, train_dl, test_dl, train_dataset, test_dataset] = get_dataset(embedding_idx, parser,
                                                                                 data_fname=data_path)
    # Printing the model and the hyper-parameters.
    if True:  # TODO-replace with condition.
        print_detail(parser)
    # creating the model according the parser.
    #  create_model(parser)
    set_datasets_measurements(the_datasets, Measurements, parser, parser.model)
    cudnn.benchmark = True  # TODO:understand what it is.
    # Loading a pretrained model if exists.
    if path_loading is not None and parser.load_model_if_exists == False:
        model_path = os.path.join(results_dir,path_loading)
        load_model(parser, model_path);
    # Deciding which parameters will be trained: if True all the model otherwise,only the task embedding.
    if train_all_model:
        learned_params = parser.model.parameters()

    if task_embedding:
        learned_params = parser.model.module.task_embedding[embedding_idx]
    if transfer_learning:
        learned_params = parser.model.module.transfer_learning[embedding_idx]
    # Training the learned params of the model.
    train_model(parser, the_datasets, learned_params, embedding_idx)
    print(accuracy(parser, test_dl))
    visualize(parser, train_dataset)


def main():
    train_omniglot(embedding_idx = 0, flag_at = FlagAt.SF, processed_data = '6_extended_Digits',
                   raw_data = '/home/sverkip/data/omniglot/data/omniglot_all_languages',model_name = '4R', path_loading = None,  train_all_model = True, train_arg = False,    task_embedding = False, transfer_learning = False,
                   load_model_if_exists = False)

main()

# Temp place
##############################################
# /home/sverkip/data/Omniglot/data/new_samples/T
# print(num_params(learned_params))
# print(accuracy_one_language(args.model, args.test_dl))
# /home/sverkip/data/Omniglot/data/results/DS=Four_languages_embedding_idx=029.06.2022 11:55:49/model5.pt
# 'DS=8_extended_exp[27, 5, 42]_embedding_idx=029.06.2022 15:21:58'
###############################################
