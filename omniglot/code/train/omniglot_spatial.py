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


def train_omniglot(embedding_idx, flag_at, processed_data, raw_data,stages, path_loading=None, training_option = 'All'):
    """
    Args:
        embedding_idx: The embedding index.
        flag_at: The model flag.
        processed_data: The samples we study from.
        raw_data: The raw_data path.
        path_loading: The path to load a pretrained model from.
        training_option:  The parameter training option
        stages: The stages we desire to learn.

    """
    # Getting the options for creating the model and the hyper-parameters.
    if  training_option not in ['all','task','arg','head']:
        raise Exception("The parameter training options are 'all', 'tasks', 'arg', 'head' ")

    parser = GetParser(flag_at, raw_data, processed_data, embedding_idx,1,stages)
    # Getting the dataset for the training.
    data_path = os.path.join(parser.data_dir, processed_data)
    [the_datasets, _ , test_dl, train_dataset, _ ] = get_dataset(parser, embedding_idx, data_fname = data_path)
    print_detail(parser)
    # creating the model according the parser.
    set_datasets_measurements(the_datasets, Measurements, parser, parser.model)
    cudnn.benchmark = True  # TODO:understand what it is.
    # Loading a pretrained model if exists.
    if path_loading is not None and parser.load_model_if_exists == False:
        model_path = os.path.join(parser.results_dir,path_loading)
        load_model(parser, model_path);
    # Deciding which parameters will be trained: if True all the model otherwise,only the task embedding.
    if training_option == 'all':
        learned_params = parser.model.parameters()
    if training_option == 'task':
        learned_params = parser.model.module.task_embedding[embedding_idx]
    if training_option == 'arg':
        learned_params = parser.model.module.arg_learning[embedding_idx]
    if training_option == 'head':
        learned_params = parser.model.module.Head_learning[embedding_idx]
    # Training the learned params of the model.
    train_model(parser, the_datasets, learned_params, embedding_idx)
    if True:
     print(accuracy(parser, test_dl))
     visualize(parser, train_dataset)

def main():
    train_omniglot(embedding_idx = 0, flag_at = FlagAt.SF, processed_data = '4L',
                raw_data = '/home/sverkip/data/BU-TD/omniglot/data/omniglot_all_languages', path_loading = None,  training_option = 'all', stages = [1])

main()
