import os
import argparse
from torch import device
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
# import supplmentery
from supplmentery import *
from supplmentery.Parser import *
from supplmentery.FlagAt import FlagAt
from supplmentery.get_dataset import get_dataset
from Configs.Config import Config
# from {Package.module} import {class}

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
dev = device("cuda") if torch.cuda.is_available() else device("cpu")
# this is not used here


def train_emnist(embedding_idx=0, flag_at=FlagAt.SF,
                 processed_data='6_extended_digits', path_loading=None,  train_all_model=True):
    # add some training options from config file
    config: Config = Config()

    cfg.gpu_interactive_queue = config.Visibility.interactive_session
    if config.Visibility.interactive_session:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # Getting the options for creating the model and the hyper-parameters.
    results_dir = '../data/emnist/data/results'
    parser = GetParser(flag_at, processed_data, embedding_idx, results_dir)
    # Getting the dataset for the training.
    data_path = os.path.join(
        '../data/new_samples/6_extended_[17]', processed_data)
    [the_datasets, train_dl, test_dl, val_dl, train_dataset,
        test_dataset] = get_dataset(embedding_idx, parser, data_fname=data_path)
    # Printing the model and the hyper-parameters.
    if True:  # TODO-replace with condition.
        print_detail(parser)
    # creating the model according the parser.
    #  create_model(parser)
    set_datasets_measurements(the_datasets, Measurements, parser, parser.model)
    cudnn.benchmark = True  # TODO:understand what it is.
    # Loading a pretrained model if exists.
    if path_loading is not None:
        load_model(parser, results_dir, path_loading)
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
    train_emnist(embedding_idx=0, flag_at=FlagAt.SF,
                 processed_data='6_extended_digits', path_loading=None,  train_all_model=True)


if __name__ == "__main__":
    main()

# Temp place
##############################################
# /home/sverkip/data/Omniglot/data/new_samples/T
# print(num_params(learned_params))
#
# /home/sverkip/data/Omniglot/data/results/DS=Four_languages_embedding_idx=029.06.2022 11:55:49/model5.pt
# 'DS=8_extended_exp[27, 5, 42]_embedding_idx=029.06.2022 15:21:58'
###############################################
