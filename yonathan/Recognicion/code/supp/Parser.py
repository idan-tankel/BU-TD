import argparse
import os
from datetime import datetime
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from supp.Dataset_and_model_type_specification import Flag, DsType, AllOptions, inputs_to_struct
from supp.batch_norm import BatchNorm
from supp.blocks import BasicBlockTD, BasicBlockBU, BasicBlockBUShared
<<<<<<< HEAD
from supp.loss_and_accuracy import UnifiedCriterion
from supp.models import BUTDModelShared, ResNet


def GetParser(task_idx=0, direction_idx = 0, model_type=BUTDModelShared, flag=Flag.NOFLAG, ds_type=DsType.Emnist, Begin_with_pretrained_model = False, use_lateral_bu_td = True,use_lateral_td_bu = True):
    """
    Args:
        task_idx: The language index.
        direction: The direction index.
        model_type: The model type.
        flag: The model flag.
        ds_type: The ds type e.g. emnist, fashionmnist, omniglot.

    Returns: A parser
    """
    if flag is not Flag.NOFLAG and model_type is ResNet:
        raise Exception("Pure ResNet can be used only in NO-FLAG mode.")
    Data_obj = AllOptions(ds_type=ds_type, flag_at=flag, ndirections=4)
    parser = argparse.ArgumentParser()
    # Flags.
    parser.add_argument('--ds_type', default = ds_type, type=DsType, help='Flag that defines the data-set type')
    parser.add_argument('--model_flag', default=flag, type=Flag, help='Flag that defines the model type')
    # Optimization arguments.
    parser.add_argument('--wd', default=0.00001, type=float, help='The weight decay of the Adam optimizer')
    parser.add_argument('--SGD', default=False, type=bool, help='Whether to use SGD or Adam optimizer')
    parser.add_argument('--initial_lr', default=0.001, type=float, help='Base lr for the SGD optimizer')
    parser.add_argument('--cycle_lr', default=True, type=bool, help='Whether to cycle the lr')
    parser.add_argument('--base_lr', default=0.0002, type=float, help='Base lr of the cyclic Adam optimizer')
    parser.add_argument('--max_lr', default=0.002, type=float,  help='Max lr of the cyclic Adam optimizer before the lr returns to the base_lr')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum of the Adam optimizer')
    parser.add_argument('--bs', default=10, type=int, help='The training batch size')
    parser.add_argument('--EPOCHS', default=100, type=int, help='Number of epochs in the training')
    # Model architecture arguments.
    parser.add_argument('--td_block_type', default=BasicBlockTD, type=nn.Module, help='Basic TD block')
    parser.add_argument('--bu_block_type', default=BasicBlockBU, type=nn.Module, help='Basic BU block')
    parser.add_argument('--bu_shared_block_type', default=BasicBlockBUShared, type=nn.Module, help='Basic shared BU block')
    parser.add_argument('--activation_fun', default=nn.ReLU, type=nn.Module, help='The non linear activation function')
    parser.add_argument('--norm_layer', default=BatchNorm, type=nn.Module, help='The used batch normalization')
    parser.add_argument('--use_lateral_butd', default=use_lateral_bu_td, type=bool, help='Whether to use the lateral connection from bu1 to TD')
    parser.add_argument('--use_lateral_tdbu', default=use_lateral_td_bu, type=bool, help='Whether to use the lateral connection from TD to BU2')
    parser.add_argument('--nfilters', default=[64, 96, 128, 256], type=list, help='The ResNet filters')
    parser.add_argument('--strides', default=[2, 2, 1, 2], type=list, help='The ResNet strides')
    parser.add_argument('--ks', default=[7, 3, 3, 3], type=list, help='The ResNet kernel sizes')
    parser.add_argument('--ns', default=[0, 3, 3, 3], type=list, help='Number of blocks per filter size')
    parser.add_argument('--nclasses', default=Data_obj.data_obj.nclasses, type=list, help='The sizes of the Linear layers')
    parser.add_argument('--ntasks', default=Data_obj.data_obj.ntasks, type=int, help='Number of tasks the model should handle')
    parser.add_argument('--ndirections', default=Data_obj.data_obj.ndirections, type = int, help='Number of directions the model should handle')
    parser.add_argument('--inshape', default=(3, 112, 224), type=tuple, help='The input image shape')
    # Training and evaluation arguments.
    parser.add_argument('--use_bu1_loss', default=Data_obj.data_obj.use_bu1_loss, type=bool, help='Whether to use the binary classification loss at the end of the BU1 stream')
    parser.add_argument('--use_bu2_loss', default=True, type=bool, help='Whether to use the classification loss at the end of the BU2 stream')
    parser.add_argument('--bu1_loss', default=nn.BCEWithLogitsLoss(reduction='mean'), type=nn.Module, help='The loss used at the end of the bu1 stream')
    parser.add_argument('--bu2_loss', default=Data_obj.data_obj.bu2_loss, type=Callable,help='The bu2 classification loss')
    parser.add_argument('--inputs_to_struct', default=inputs_to_struct, help='')
    parser.add_argument('--criterion', default = UnifiedCriterion, type=Callable,  help='The unified loss function of all training')
    parser.add_argument('--task_accuracy', default=Data_obj.data_obj.task_accuracy, type=Callable,  help='The accuracy function')
    # Data arguments.
    parser.add_argument('--device', default =  torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"), help = '')
    parser.add_argument('--workers', default=2, type=int, help='Number of workers to use')
    # Training procedure.
    parser.add_argument('--train_arg_emb', default=Data_obj.data_obj.train_arg_emb, type=bool, help='Flag whether to add ReLU in certain places')
    parser.add_argument('--initial_tasks', default=Data_obj.data_obj.initial_tasks, type=list, help='The initial tasks to train first')
    #
    # Change to another function
    now = datetime.now()
    dt_string = now.strftime("%d.%m.%Y %H:%M:%S")
    model_path = "Model" + str(task_idx) + '_' + str(direction_idx) + "_test_stronger_emb" + dt_string
    parser.add_argument('--results_dir', default=Data_obj.data_obj.results_dir, type=str, help='The direction the model will be stored')
    model_dir = os.path.join(Data_obj.data_obj.results_dir, model_path)
    parser.add_argument('--model_dir', default=model_dir, type=str, help='The direction the model will be stored')

    parser.add_argument('--model', default=model_type(parser.parse_args()).cuda(), type=nn.Module,  help='The model we fit')
    ########################################
    # For Baselines only.
    ########################################
    lambdas_LWF = [0.2, 0.1, 0.05, 0.025, 0.0125 ,0.125/2.0, 0.3]
    # EWC
    parser.add_argument('--ewc_lambda', default=1e10, type = float, help='The ewc strength')
    # SI
    parser.add_argument('--si_lambda', default=1000, type = float, help='The ewc strength')
    parser.add_argument('--si_eps', default = 0.0000001, type=float, help='The ewc strength')
    # LFL
    parser.add_argument('--lambda_lfl', default=1e5, type = float, help='The ewc strength')
    # LWF
    parser.add_argument('--alpha_LWF', default = lambdas_LWF[6], type = float, help='The ewc strength')
=======
from supp.data_functions import dev
from supp.loss_and_accuracy import UnifiedLossFun
from supp.models import BUTDModelShared


def GetParser(language_idx=0, direction='right', model_type=BUTDModelShared, flag=Flag.NOFLAG, ds_type=DsType.Emnist,
              Begin_with_pretrained_model=False):
    """
    Args:
        language_idx: The language index.
        direction: The direction.
        model_type: The model type.
        flag: The flag.
        ds_type:
        Begin_with_pretrained_model:

    Returns:

    """
    # TODO -  ADD SOME ASSERTS TO AVOID CONFLICTS.
    parser = argparse.ArgumentParser()
    num_gpus = torch.cuda.device_count()
    Data_obj = AllOptions(ds_type = ds_type,flag_at = flag,ntasks = 4)

    parser.add_argument('--generalize', default=Data_obj.data_obj.generelize, type=bool, help='Whether to test also the CG test')
    parser.add_argument('--ds_type', default = ds_type, type=DsType, help='Flag that defines the data-set type')
    parser.add_argument('--wd', default=0.00001, type=float, help='The weight decay of the Adam optimizer')
    parser.add_argument('--SGD', default=False, type=bool, help='Whether to use SGD or Adam optimizer')
    parser.add_argument('--initial_lr', default=0.001, type=float, help='Base lr for the SGD optimizer')
    parser.add_argument('--ndirections', default=Data_obj.data_obj.ndirections, type=int,
                        help='Number of directions the model should handle')
    parser.add_argument('--checkpoints_per_epoch', default=1, type=int, help='Number of model saves per epoch')
 #   parser.add_argument('--use_td_flag', default=opts.use_td_flag, type=bool, help='Whether to use the td flag')  #
    parser.add_argument('--initial_tasks', default=Data_obj.data_obj.initial_tasks, type=list,
                        help='The initial tasks to train first')
    parser.add_argument('--bs', default=10, type=int, help='The training batch size')
    parser.add_argument('--scale_batch_size', default=num_gpus * 10, type=int, help='scale batch size')
    parser.add_argument('--saving_metric', default='accuracy', type=str,
                        help='The metric to save models according to')  # TODO-understand what is distributed.
    parser.add_argument('--workers', default=2, type=int, help='Number of workers to use')
    parser.add_argument('--cycle_lr', default=True, type=bool, help='Whether to cycle the lr')
    parser.add_argument('--orig_relus', default=False, type=bool, help='Flag whether to add ReLU in certain places')
    # TODO - DELETE ORIG_RELUS.
    parser.add_argument('--base_lr', default=0.0002, type=float, help='Base lr of the cyclic Adam optimizer')
    parser.add_argument('--max_lr', default=0.002, type=float,
                        help='Max lr of the cyclic Adam optimizer before the lr returns to the base_lr')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum of the Adam optimizer')
    parser.add_argument('--model_flag', default = flag, type=Flag, help='Flag that defines the model type')
    parser.add_argument('--ntasks', default = Data_obj.data_obj.ntasks, type=int, help='Number of tasks the model should handle')
    #   parser.add_argument('--nargs', default=6, type=int, help='Number of possible position arguments in the image')
    parser.add_argument('--activation_fun', default=nn.ReLU, type=nn.Module, help='Non Linear activation function')
    parser.add_argument('--use_bu1_loss', default = Data_obj.data_obj.use_bu1_loss, type=bool,
                        help='Whether to use the binary classification loss at the end of the BU1 stream')
    parser.add_argument('--use_bu2_loss', default=True, type=bool,
                        help='Whether to use the classification loss at the end of the BU2 stream')
    parser.add_argument('--use_lateral_butd', default=True, type=bool,
                        help='Whether to use the lateral connection from bu1 to TD')
    parser.add_argument('--use_lateral_tdbu', default=True, type=bool,
                        help='Whether to use the lateral connection from TD to BU2')
    parser.add_argument('--nfilters', default=[64, 96, 128, 256], type=list, help='The ResNet filters')
    parser.add_argument('--strides', default=[2, 2, 1, 2], type=list, help='The ResNet strides')
    parser.add_argument('--ks', default=[7, 3, 3, 3], type=list, help='The ResNet kernel sizes')
    parser.add_argument('--ns', default=[0, 1, 1, 1], type=list, help='Number of blocks per filter size')
    parser.add_argument('--inshape', default=(3, 112, 224), type=tuple, help='The input image shape')
    parser.add_argument('--norm_layer', default=BatchNorm, type=nn.Module, help='The used batch normalization')
    parser.add_argument('--EPOCHS', default=100, type=int, help='Number of epochs in the training')
    parser.add_argument('--num_gpus', default=num_gpus, type=int, help='number of used gpus')
    # Change to another function
    now = datetime.now()
    dt_string = now.strftime("%d.%m.%Y %H:%M:%S")
    model_path = "Model" + str(language_idx) + '_' + str(direction) + "_test_stronger_emb" + dt_string
    parser.add_argument('--results_dir', default=Data_obj.data_obj.results_dir, type=str, help='The direction the model will be stored')
    model_dir = os.path.join(Data_obj.data_obj.results_dir, model_path)
    parser.add_argument('--model_dir', default=model_dir, type=str, help='The direction the model will be stored')
    parser.add_argument('--nclasses', default=Data_obj.data_obj.nclasses, type=list, help='The sizes of the Linear layers')
    parser.add_argument('--logfname', default='log.txt', type=str, help='The name of the log file')
    parser.add_argument('--bu2_loss', default=Data_obj.data_obj.bu2_loss, type=Callable, help='The bu2 classification loss')
    parser.add_argument('--task_accuracy', default=Data_obj.data_obj.task_accuracy, type=Callable, help='The accuracy function')
    parser.add_argument('--nheads', default=1, type = int, help='Needs to be derivatived from the Flag')
    parser.add_argument('--learning_rates_mult', default = np.ones(100), type=list, help='The scaled learning rated')
    parser.add_argument('--save_model', default = True, type=bool, help='Whether to save the model on the disk')
    parser.add_argument('--td_block_type', default = BasicBlockTD, type=nn.Module, help='Basic TD block')
    parser.add_argument('--bu_block_type', default = BasicBlockBU, type=nn.Module, help='Basic BU block')
    parser.add_argument('--inputs_to_struct', default = inputs_to_struct, help='')
    parser.add_argument('--bu_shared_block_type', default=BasicBlockBUShared, type=nn.Module,
                        help='Basic shared BU block')
    parser.add_argument('--bu1_loss', default=nn.BCEWithLogitsLoss(reduction='mean').to(dev), type=nn.Module,
                        help='The loss used at the end of the bu1 stream')
    parser.add_argument('--bu2_classification_loss', default=nn.CrossEntropyLoss(reduction='none').to(dev),
                        type=nn.Module, help='The loss used at the end of the bu2 stream')
    parser.add_argument('--model', default=model_type(parser.parse_args()).cuda(), type=nn.Module,
                        help='The model we fit')
    parser.add_argument('--criterion', default=UnifiedLossFun(parser.parse_args()), type=Callable,
                        help='The unified loss function of all training')
    parser.add_argument('--epoch_save_idx', default='accuracy', type=str,
                        help='The metric we update the best model according to(usually loss/accuracy)')
    parser.add_argument('--dataset_saving_by', default = Data_obj.data_obj.dataset_saving_by, type=str,
                        help='The dataset we update the best model according to(usually val/test)')
    #
    # For Baselines only.
    # EWC
    parser.add_argument('--ewc_lambda', default=1e8, type = float, help='The ewc strength')
    # SI
    parser.add_argument('--si_lambda', default=0.0, type = float, help='The ewc strength')
    # LFL
    parser.add_argument('--lambda_lfl', default=1e0 / 1.0, type = float, help='The ewc strength')
    # LWF
    parser.add_argument('--alpha_LWF', default=100 * 20.0, type = float, help='The ewc strength')
>>>>>>> 315b11ac3016dc72662fd8ca96881ae68c5cda6d
    parser.add_argument('--temperature_LWF', default=2.0, type = float, help='The ewc strength')
    # MAS
    parser.add_argument('--mas_alpha', default=100 * 20.0, type=float, help='The ewc strength')
    parser.add_argument('--mas_lambda_reg', default=2, type=float, help='The ewc strength')
<<<<<<< HEAD
    # RWALK
   # ewc_lambda: float = 0.1, ewc_alpha: float = 0.9, delta_t: int = 10
    parser.add_argument('--rwalk_ewc_lambda', default = 1e10, type=float, help='The ewc strength')
    parser.add_argument('--rwalk_ewc_alpha', default=0.9, type=float, help='The ewc strength')
    parser.add_argument('--rwalk_delta_t', default=10, type=int, help='The ewc strength')
    #
    # General arguments.
=======
    #
>>>>>>> 315b11ac3016dc72662fd8ca96881ae68c5cda6d
    parser.add_argument('--train_mb_size', default=10, type=int, help='The ewc strength')
    parser.add_argument('--eval_mb_size', default=10, type=int, help='The ewc strength')
    parser.add_argument('--train_epochs', default=1, type=int, help='The ewc strength')
    parser.add_argument('--epochs', default=20, type=int, help='The ewc strength')
    parser.add_argument('--pretrained_model', default=Begin_with_pretrained_model, type=int, help='The ewc strength')
<<<<<<< HEAD
=======

>>>>>>> 315b11ac3016dc72662fd8ca96881ae68c5cda6d
    return parser.parse_args()
