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
from supp.data_functions import dev
import torch.nn as nn
# NO SEED in data_functions and not in blocks.
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Training_flag:
    def __init__(self,train_all_model, train_arg,task_embedding,head_learning):
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
        if self.task_embedding:
            learned_params.extend(model.module.task_embedding[direction])
        if self.head_learning:
            learned_params.extend(model.module.transfer_learning[lang_idx][direction])
        if self.train_arg:
            learned_params.extend(model.module.tdmodel.argument_embedding[lang_idx])
        return learned_params

def train_omniglot(opts:argparse, lang_idx:int, the_datasets:list, training_flag:Training_flag, direction:int):
    """
    Args:
        opts: The model options.
        lang_idx: The language index.
        the_datasets: The datasets.
        training_flag: The training flag.
        direction: The direction.

    Returns:

    """
    set_datasets_measurements(the_datasets, Measurements, opts, opts.model)
    cudnn.benchmark = True  # TODO:understand what it is.
    # Deciding which parameters will be trained: if True all the model otherwise,only the task embedding.
    learned_params =training_flag.Get_learned_params(opts.model, lang_idx, direction)
    opts.optimizer, opts.scheduler = create_optimizer_and_sched(opts, learned_params)
    # Training the learned params of the model.
    fit(opts, the_datasets, lang_idx, direction)

def main_omniglot(language_idx,train_right,train_left):
    """
    Args:
        language_idx:
        train_right:
        train_left:

    Returns:
    """
    opts = Model_Options_By_Flag_And_DsType(Flag=Flag.SF, DsType=DsType.Omniglot)
    parser = GetParser(opts=opts, language_idx=0)
    print_detail(parser)
    embedding_idx = 0
    data_path = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/omniglot/samples/6_extended_testing_new_changes_beta_5R'
    # Create the data for right.
    [the_datasets, _,  test_dl, _ , _ , _, _] = get_dataset_for_spatial_realtions(parser, data_path,lang_idx = 0, direction =0)
    # Training Right.
    path_loading = '5R_new_version/model_latest_right.pt'

    model_path = parser.results_dir

    load_model(parser, model_path, path_loading, load_optimizer_and_schedular=False);
  #  load_running_stats(parser.model, task_emb_id = 0);
  #  acc = accuracy(parser, test_dl)
    #print("Done training right, with accuracy : " + str(acc))
    if train_right:
        parser.EPOCHS = 60
        training_flag = Training_flag(train_all_model=True, train_arg=True, task_embedding=False, head_learning=True)
        train_omniglot(parser, lang_idx=0, the_datasets=the_datasets, training_flag=training_flag)

    if train_left:
        parser.EPOCHS = 100
        [the_datasets, _, _, _, _, _, _] = get_dataset_for_spatial_realtions(parser, data_path, lang_idx = 0, direction = 1 )
        training_flag = Training_flag(train_all_model=False, train_arg=False, task_embedding=True, head_learning=True)
        train_omniglot(parser, lang_idx= 0, the_datasets=the_datasets, training_flag=training_flag, direction = 1)

main_omniglot(24,False,True)