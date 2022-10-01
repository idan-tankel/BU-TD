import os
import argparse
import wandb
import torch.backends.cudnn as cudnn
from supp.Parser import GetParser
from supp.get_dataset import get_dataset_for_spatial_realtions
from supp.FlagAt import Flag, DsType,Model_Options_By_Flag_And_DsType
from supp.logger import print_detail
from supp.loss_and_accuracy import accuracy
from supp.measurments import set_datasets_measurements
from supp.training_functions import load_model, fit
from supp.general_functions import num_params, create_optimizer_and_sched
from supp.measurments import Measurements
from supp.batch_norm import load_running_stats
import argparse

import torch.nn as nn
# NO SEED in data_functions and not in blocks.
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Training_flag:
    def __init__(self,train_all_model:bool, train_arg:bool,task_embedding:bool,head_learning:bool):
        """
        Args:
            train_all_model: Whether to train all model.
            train_arg: Whether to train arg.
            task_embedding: Whether to train the task embedding.
            head_learning: Whether to train the read out head.
        """
        self.train_all_model = train_all_model
        self.train_arg = train_arg
        self.task_embedding = task_embedding
        self.head_learning = head_learning

    def Get_learned_params(self, model:nn.Module, lang_idx:int, direction:int):
        """
        Args:
            model: The model.
            lang_idx: Language index.
            direction: The direction.

        Returns: The desired parameters.

        """
        if self.train_all_model:
            return list(model.parameters())
        learned_params = []
        idx = direction + 4 * lang_idx
        if self.task_embedding:
            learned_params.extend(model.module.task_embedding[direction])
        if self.head_learning:
            learned_params.extend(model.module.transfer_learning[idx])
        if self.train_arg:
            learned_params.extend(model.module.argument_embedding[lang_idx])
        return learned_params

def train_omniglot(opts:argparse, lang_idx:int, the_datasets:list, training_flag:Training_flag, direction:int):
    """
    Args:
        opts: The model options.
        lang_idx: The language index.
        the_datasets: The datasets.
        training_flag: The training flag.
        direction: The direction.

    Returns: The optimum and the learning history.

    """
    set_datasets_measurements(the_datasets, Measurements, opts, opts.model)
    cudnn.benchmark = True  # TODO:understand what it is.
    # Deciding which parameters will be trained: if True all the model otherwise,only the task embedding.
    learned_params =training_flag.Get_learned_params(opts.model, lang_idx, direction)
    opts.optimizer, opts.scheduler = create_optimizer_and_sched(opts, learned_params)
    # Training the learned params of the model.
    return fit(opts, the_datasets, lang_idx, direction)


def name(index):
    if index ==-1:
        return "5R"
    else:
     return "6_extended_"+str(index)


def main_omniglot(lang_idx:int=-1,train_right:bool = True,train_left:bool = True):
    """
    Args:
        lang_idx:
        train_right: Whether to train right.
        train_left: Whether to train left.

    Returns: None.
    """
    opts = Model_Options_By_Flag_And_DsType(Flag=Flag.ZF, DsType=DsType.Omniglot)
    parser = GetParser(opts=opts, language_idx=lang_idx,direction = 'left')
    print_detail(parser)
    data_path = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/omniglot/samples_new/'+name(lang_idx)
    # Create the data for right.
    [the_datasets, _ ,  test_dl, _ ,] = get_dataset_for_spatial_realtions(parser, data_path,lang_idx = lang_idx + 1, direction = 0)
    # Training Right.
    path_loading = os.path.join('Model{}_right'.format(lang_idx), 'model_right_best.pt')
    model_path = parser.results_dir
    load_model(parser.model, model_path, path_loading, load_optimizer_and_schedular=False);
    #load_running_stats(parser.model, task_emb_id = 1);
  #  acc = accuracy(parser.model, test_dl)
 #   print("Done training right, with accuracy : " + str(acc))
    if train_right:
        parser.EPOCHS = 60
        training_flag = Training_flag(train_all_model=False, train_arg=True, task_embedding=False, head_learning=True)
        train_omniglot(parser, lang_idx = lang_idx +1, the_datasets=the_datasets, training_flag=training_flag, direction = 0)

    if train_left:
        parser.EPOCHS = 100
        [the_datasets, _ , _ , _ ] = get_dataset_for_spatial_realtions(parser, data_path, lang_idx = lang_idx + 1, direction = 1 )
        training_flag = Training_flag(train_all_model = False, train_arg=False, task_embedding = False, head_learning = True)
        train_omniglot(parser, lang_idx = lang_idx + 1, the_datasets = the_datasets, training_flag = training_flag, direction = 1)

main_omniglot(48,False,True)