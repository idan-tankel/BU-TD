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

def BN(model: nn.Module) :
    """
    loads the running_stats of the task_id for each norm_layer.
    Args:
        model: The model we want to load its variables.
        task_emb_id: the task_id

    """
    sum =0.0
    for _, layer in model.named_modules():
        if isinstance(layer, BatchNorm):
           sum+= num_params(layer.parameters())
    return sum

def train_omniglot(embedding_idxs, processed_data,stages, path_loading=None, training_option = 'All',freeze = False):
    """
    Args:
        embedding_idx: The embedding index.
        flag_at: The model flag.
        processed_data: The samples we study from.
        path_loading: The path to load a pretrained model from.
        training_option:  The parameter training option
        stages: The stages we desire to learn.

    """
    # Getting the options for creating the model and the hyper-parameters.
    if  training_option not in ['all','task','arg','head']:
        raise Exception("The parameter training options are 'all', 'tasks', 'arg', 'head' ")
    embedding_idx = embedding_idxs[1]
    parser = GetParser(processed_data, embedding_idxs, stages, 1)
    parser.freeze = freeze
    # Getting the dataset for the training.
    data_path = os.path.join(parser.data_dir, processed_data)
    [the_datasets, train_dl , test_dl, train_dataset, _ ] = get_dataset(parser, embedding_idxs, data_fname = data_path)
    print_detail(parser)
    parser.test_dl = test_dl
    # creating the model according the parser.
    set_datasets_measurements(the_datasets, Measurements, parser)
    cudnn.benchmark = True  # TODO:understand what it is.
    # Loading a pretrained model if exists.
    if path_loading is not None:
        model_path = os.path.join(parser.results_dir,path_loading)
        load_model(parser, model_path);
    # Deciding which parameters will be trained: if True all the model otherwise,only the task embedding.
    if training_option == 'all':
        learned_params = parser.model.parameters()
    if training_option == 'task':
       learned_params = parser.model.module.task_embedding[embedding_idx]
       learned_params.extend(parser.model.module.Head_learning[embedding_idx][0])
       learned_params.extend(parser.model.module.Head_learning[embedding_idx][1])
       learned_params.extend(parser.model.module.Head_learning[embedding_idx][2])
       #learned_params = parser.model.module.task_embedding[40]
    if training_option == 'arg':
        learned_params = parser.model.module.arg_learning[embedding_idx]
    if training_option == 'head':
        learned_params = parser.model.module.Head_learning[embedding_idx][1]
    # Training the learned params of the model.
  # train_model(parser, the_datasets, learned_params, embedding_idxs[1])
    print(num_params(learned_params))
    print(BN(parser.model))
    print(accuracy(parser, test_dl, stages))
    # visualize(parser, train_dataset)

def main():
    train_omniglot(embedding_idxs = [0, 0, 0], processed_data = '4_test_offsets_110K[49]', path_loading = None
    , training_option = 'task', stages = [0,1,2], freeze = True )

main()
#