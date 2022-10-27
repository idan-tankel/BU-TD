import argparse
import os
from pathlib import Path

import torch.backends.cudnn as cudnn
import torch.nn as nn

from supp.Dataset_and_model_type_specification import Flag
from supp.Parser import GetParser
from supp.utils import create_optimizer_and_scheduler
from supp.get_dataset import get_dataset_for_spatial_realtions
from supp.logger import print_detail
from supp.loss_and_accuracy import accuracy
from supp.measurments import Measurements
from supp.measurments import set_datasets_measurements
from supp.models import BUTDModelShared
from supp.training_functions import fit

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cudnn.benchmark = True

class Training_flag:
    def __init__(self, train_all_model: bool, train_arg: bool, task_embedding: bool, head_learning: bool):
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

    def Get_learned_params(self, model: nn.Module, lang_idx: int, direction: int):
        """
        Args:
            model: The model.
            lang_idx: Language index.
            direction: The direction.

        Returns: The desired parameters.

        """
        if self.train_all_model:
            return list(model.parameters())
        idx = lang_idx * 4 + direction
        learned_params = []
        if self.task_embedding:
            learned_params.extend(model.task_embedding[direction])
        if self.head_learning:
            learned_params.extend(model.transfer_learning[idx])
        if self.train_arg:
            learned_params.extend(model.tdmodel.argument_embedding[lang_idx])
        return learned_params


def train_omniglot(opts: argparse, lang_idx: int, the_datasets: list, training_flag: Training_flag, direction: int):
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
    learned_params = training_flag.Get_learned_params(opts.model, lang_idx, direction)
    opts.optimizer, opts.scheduler = create_optimizer_and_scheduler(opts, learned_params)
    # Training the learned params of the model.
    return fit(opts, the_datasets, lang_idx, direction)

def main_emnist(train_right: bool, train_left: bool,direction:int):
    """
    Args:
        train_right: Whether to train right.
        train_left: Whether to train left.

    Returns: None.
    """
    parser = GetParser(language_idx=0,model_type=BUTDModelShared,flag=Flag.TD)
    print_detail(parser)
    project_path = Path(__file__).parents[1]
    data_path =  os.path.join(project_path, 'data/{}/samples/6_extended'.format('emnist'))
    # Create the data for right.
    [the_datasets, train_dl, test_dl, _ , _, _, _] = get_dataset_for_spatial_realtions(parser, data_path, lang_idx=0, direction = 0)
    # Training Right.
    path_loading = 'Model_up_1/model_right_best.pt'

    model_path = parser.results_dir
   # load_model(parser.model, model_path, path_loading, load_optimizer_and_schedular=False);
    #   load_running_stats(parser.model, task_emb_id = 0,direction_id =0);
    acc = accuracy(parser, test_dl)
    print("Done training right, with accuracy : " + str(acc))
  #  print(num_params(parser.model.parameters()))
    if train_right:
        parser.EPOCHS = 60
        training_flag = Training_flag(train_all_model=True, train_arg=True, task_embedding=False, head_learning=True)
        train_omniglot(parser, lang_idx=0, the_datasets=the_datasets, training_flag=training_flag, direction=0)

    if train_left:
     #   load_model(parser.model_old, model_path, path_loading, load_optimizer_and_schedular=False)
        #    parser.model_old = copy.deepcopy(parser.model)
        # parser.model_old = copy.deepcopy(parser.model_old)
        # acc = accuracy(parser.model_old, test_dl)
        parser.EPOCHS = 100
        [the_datasets, _ , _ , _, train_ds, test_ds , _ ] = get_dataset_for_spatial_realtions(parser, data_path, lang_idx=0, direction=direction)
#        compute_for_each_direction_the_stats(train_ds)
        training_flag = Training_flag(train_all_model=False, train_arg=False, task_embedding=False, head_learning=True)
        train_omniglot(parser, lang_idx=0, the_datasets=the_datasets, training_flag=training_flag, direction=direction)


main_emnist(True, True, direction = 0)
