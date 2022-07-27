import os
import argparse
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import sys
sys.path.append("..")
# import supplmentery
from supplmentery import *
from supplmentery.Parser import *
from supplmentery.FlagAt import FlagAt
# from utils.funcs import *
# from supplementery.Parser import *
# from supplementery.get_dataset import *
# from supplementery.create_model import *
# from supplementery.FlagAt import *
# from supplementery.emnist_dataset import inputs_to_struct as inputs_to_struct
# from supplementery.logger import *
# from supplementery.data_functions import *
# from supplementery.loss_and_accuracy import *
# from supplementery.visuialize_predctions import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def train_emnist(embedding_idx, flag_at, processed_data, path_loading=None, train_all_model=True):
    # Getting the options for creating the model and the hyper-parameters.
    results_dir = '/home/sverkip/data/emnist/data/results'
    parser = GetParser(flag_at, processed_data, embedding_idx, results_dir)
    # Getting the dataset for the training.
    data_path = os.path.join('/home/sverkip/data/Emnist_NEW_version/data/samples/', processed_data)
    [the_datasets, train_dl, test_dl, val_dl, train_dataset, test_dataset] = get_dataset(embedding_idx, parser, data_fname=data_path)
    # Printing the model and the hyper-parameters.
    if True:  # TODO-replace with condition.
        print_detail(parser)
    # creating the model according the parser.
    #  create_model(parser)
    set_datasets_measurements(the_datasets, Measurements, parser, parser.model)
    cudnn.benchmark = True  # TODO:understand what it is.
    # Loading a pretrained model if exists.
    if path_loading is not None:
        load_model(parser, results_dir, path_loading);
    # Deciding which parameters will be trained: if True all the model otherwise,only the task embedding.
    if train_all_model:
        learned_params = parser.model.parameters()
    task_embedding = False
    transfer_learning = False
    if task_embedding:
        learned_params = parser.model.module.task_embedding[embedding_idx]
    if transfer_learning:
        learned_params = parser.model.module.transfer_learning[embedding_idx]
    # Training the learned params of the model.
    # print(accuracy(parser, val_dl))
    train_model(parser, the_datasets, learned_params, embedding_idx)
    visualize(parser, train_dataset)

def main():
    train_emnist(embedding_idx = 0, flag_at=FlagAt.SF, processed_data='6_extended_digits', path_loading=None,  train_all_model = True)

main()

# Temp place
##############################################
# /home/sverkip/data/Omniglot/data/new_samples/T
# print(num_params(learned_params))
#
# /home/sverkip/data/Omniglot/data/results/DS=Four_languages_embedding_idx=029.06.2022 11:55:49/model5.pt
# 'DS=8_extended_exp[27, 5, 42]_embedding_idx=029.06.2022 15:21:58'
###############################################
