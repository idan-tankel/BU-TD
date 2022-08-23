import os
import argparse
from torch import device
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
# import supplmentery
import supplmentery
from supplmentery import measurments, training_functions, logger, visuialize_predctions
from supplmentery.Parser import *
from supplmentery.FlagAt import FlagAt
from supplmentery.get_dataset import get_dataset
from Configs.Config import Config
# from {Package.module} import {class}


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
dev = device("cuda") if torch.cuda.is_available() else device("cpu")
# this is not used here


def train_emnist(embedding_idx=0, flag_at=FlagAt.SF,
                 processed_data='5_extended', path_loading=None,  train_all_model=True):
    """
    train_emnist This is the main training function under the main function

    Args:
        embedding_idx (int, optional): _description_. Defaults to 0.
        flag_at (_type_, optional): _description_. Defaults to FlagAt.SF.
        processed_data (str, optional): _description_. Defaults to '5_extended'.
        path_loading (_type_, optional): _description_. Defaults to None.
        train_all_model (bool, optional): _description_. Defaults to True.
    """                 
    # add some training options from config file
    config: Config = Config()

    if config.Visibility.interactive_session:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # Getting the options for creating the model and the hyper-parameters.
    results_dir = '../data/emnist/data/results'
    parser = GetParser(flag_at, processed_data, embedding_idx, results_dir)
    parser= config
    # Getting the dataset for the training.
    # TODO initialize all model options from args (see `v26.functions.inits.py` under itsik branch)


    

    data_path = os.path.join(
        '../data/new_samples', processed_data)
    [the_datasets, train_dl, test_dl, val_dl, train_dataset,
        test_dataset] = get_dataset(direction=embedding_idx, args=parser, data_fname=data_path)
    # Printing the model and the hyper-parameters.
    logger.print_detail(parser)
    # creating the model according the parser.
    model  = create_model.create_model(model_opts=parser)
    measurments.set_datasets_measurements(
        the_datasets, measurments.Measurements, parser, model=model)
    
    cudnn.benchmark = True  # TODO:understand what it is.
    # Loading a pretrained model if exists.
    if path_loading is not None:
        training_functions.load_model(parser, results_dir, path_loading)
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
    training_functions.train_model(
        parser, the_datasets, learned_params, embedding_idx)
    visuialize_predctions.visualize(parser, train_dataset)







def main():
    train_emnist(embedding_idx=0, flag_at=FlagAt.SF,
                 processed_data='5_extended', path_loading=None,  train_all_model=True)


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
