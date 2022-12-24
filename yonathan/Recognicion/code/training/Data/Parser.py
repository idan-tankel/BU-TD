import argparse
import os

from typing import Callable, Union, ClassVar

import torch
import torch.nn as nn

from Data_Creation.Create_dataset_classes import DsType  # Import the Data_Creation set types.

from training.Data.Data_params import Flag, AllDataSetOptions
from training.Data.Structs import inputs_to_struct, outs_to_struct
from training.Metrics.Loss import UnifiedCriterion
from training.Modules.Batch_norm import BatchNorm
from training.Modules.Model_Blocks import BasicBlockTD, BasicBlockBU, BasicBlockBUShared
from training.Modules.Models import BUTDModel, ResNet


def GetParser(task_idx: int = 0, direction_idx: int = 0, model_type: Union[BUTDModel, ResNet] = BUTDModel,
              model_flag: Flag = Flag.NOFLAG, ds_type: DsType = DsType.Emnist):
    """
    Args:
        task_idx: The task index.
        direction_idx: The direction index.
        model_type: The model type  BUTD or ResNet.
        model_flag: The model flag.
        ds_type: The data type e.g. mnist, fashionmnist, omniglot.

    Returns: The options parser.

    """
    # Asserting NOFLAG is used only with ResNet.
    if model_flag is not Flag.NOFLAG and model_type is ResNet:
        raise Exception("Pure ResNet can be trained only in NO-FLAG mode.")
    # Asserting the task idx is meaningful only for Omniglot.
    if task_idx != 0 and (ds_type is DsType.Emnist or ds_type is DsType.Fashionmnist):
        raise Exception("The task id is used only for Omniglot.")
    #
    Data_specification = AllDataSetOptions(ds_type=ds_type, flag_at=model_flag,
                                           initial_task_for_omniglot_only=5)
    parser = argparse.ArgumentParser()
    # Flags.
    parser.add_argument('--ds_type', default=ds_type, type=DsType, help='Flag that defines the data-set type')
    parser.add_argument('--model_flag', default=model_flag, type=Flag, help='Flag that defines the model type')
    # Optimization arguments.
    parser.add_argument('--wd', default=0.00001, type=float, help='The weight decay of the Adam optimizer')
    parser.add_argument('--SGD', default=False, type=bool, help='Whether to use SGD or Adam optimizer')
    parser.add_argument('--initial_lr', default=0.0001, type=float, help='Base lr for the SGD optimizer')
    parser.add_argument('--cycle_lr', default=True, type=bool, help='Whether to cycle the lr')
    parser.add_argument('--base_lr', default=0.0002, type=float, help='Base lr of the cyclic scheduler')
    parser.add_argument('--max_lr', default=0.002, type=float,
                        help='Max lr of the cyclic scheduler before the lr returns to the base_lr')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum of the optimizer')
    parser.add_argument('--bs', default=10, type=int, help='The training batch size')
    parser.add_argument('--EPOCHS', default=60, type=int, help='Number of epochs in the training')
    # Model architecture arguments.
    parser.add_argument('--model_type', default=model_type, type=DsType, help='The model type BUTD or ResNet')
    parser.add_argument('--td_block_type', default=BasicBlockTD, type=nn.Module, help='Basic TD block')
    parser.add_argument('--bu_block_type', default=BasicBlockBU, type=nn.Module, help='Basic BU block')
    parser.add_argument('--bu_shared_block_type', default=BasicBlockBUShared, type=nn.Module,
                        help='Basic shared BU block')
    parser.add_argument('--activation_fun', default=nn.ReLU, type=nn.Module, help='The non linear activation function')
    parser.add_argument('--norm_layer', default=BatchNorm, type=nn.Module, help='The used batch normalization')
    parser.add_argument('--use_lateral_butd', default=True, type=bool,
                        help='Whether to use the lateral connection from BU1 to TD')
    parser.add_argument('--use_lateral_tdbu', default=True, type=bool,
                        help='Whether to use the lateral connection from TD to BU2')
    parser.add_argument('--use_additional_emb', default=True, type=bool,
                        help='Whether to use additional embedding')
    parser.add_argument('--nfilters', default=[64, 96, 128, 256], type=list, help='The ResNet filters')
    parser.add_argument('--strides', default=[2, 2, 1, 2], type=list, help='The ResNet strides')
    parser.add_argument('--ks', default=[7, 3, 3, 3], type=list, help='The ResNet kernel sizes')
    parser.add_argument('--ns', default=[0, 1, 1, 1], type=list, help='Number of blocks per layer')
    parser.add_argument('--nclasses', default=Data_specification.data_obj.nclasses, type=list,
                        help='The sizes of the task-head classes')
    parser.add_argument('--ntasks', default=Data_specification.data_obj.ntasks, type=int,
                        help='Number of tasks the model should solve')
    parser.add_argument('--ndirections', default=Data_specification.data_obj.ndirections, type=int,
                        help='Number of directions the model should handle')
    parser.add_argument('--inshape', default=(3, *Data_specification.data_obj.image_size), type=tuple,
                        help='The input image shape, maybe override in get_dataset')
    parser.add_argument('--num_heads', default=Data_specification.data_obj.num_heads, type=list,
                        help='The number of headed for each task, direction')
    parser.add_argument('--num_x_axis', default=Data_specification.data_obj.num_x_axis, type=int,
                        help='The neighbor radius in the x-axis.')
    parser.add_argument('--num_y_axis', default=Data_specification.data_obj.num_y_axis, type=int,
                        help='The neighbor radius in the y-axis.')
    parser.add_argument('--shared', default=True, help='Whether the conv layers of BU1, BU2 should be identical.')
    # Training and evaluation arguments.
    parser.add_argument('--use_bu1_loss', default=Data_specification.data_obj.use_bu1_loss, type=bool,
                        help='Whether to use the binary classification loss at the end of the BU1 stream')
    parser.add_argument('--bu1_loss', default=nn.BCEWithLogitsLoss(), type=Callable,
                        help='The loss used at the end of the bu1 stream')
    parser.add_argument('--bu2_loss', default=Data_specification.data_obj.bu2_loss, type=Callable,
                        help='The bu2 classification loss')
    parser.add_argument('--criterion', default=UnifiedCriterion, type=Callable,
                        help='The unified loss function of all training')
    parser.add_argument('--task_accuracy', default=Data_specification.data_obj.task_accuracy, type=Callable,
                        help='The Accuracy function')
    # Struct variables.
    parser.add_argument('--inputs_to_struct', default=inputs_to_struct, type=ClassVar,
                        help='The struct transform the list of inputs to struct.')
    parser.add_argument('--outs_to_struct', default=outs_to_struct, type=ClassVar,
                        help='The struct transform the list of outputs to struct.')
    # Data arguments.
    parser.add_argument('--device', default=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                        type=str, help='The device we use,usually cuda')
    parser.add_argument('--workers', default=2, type=int, help='Number of workers to use')
    # Training procedure.
    parser.add_argument('--initial_directions', default=Data_specification.data_obj.initial_directions, type=list,
                        help='The initial tasks to train first')
    # Change to another function
    model_path = "Model_task_{}_direction_{}_ds_{}".format(task_idx, direction_idx, str(ds_type))
    parser.add_argument('--results_dir', default=Data_specification.data_obj.results_dir, type=str,
                        help='The path the model will be stored')
    model_dir = os.path.join(Data_specification.data_obj.results_dir, model_path)
    parser.add_argument('--model_dir', default=model_dir, type=str, help='The model path')
    ########################################
    # For Avalanche Baselines only.
    ########################################
    lambdas_LWF = [10.0, 50.0, 100.0, 200.0, 1000]
    lambda_LFL = [0.05, 0.25, 0.375, 0.2, 0.5, 0.8]
    lambda_rwalk = [20.0, 30.0, 40.0, 50.0, 10.0, 5.0, 2.5]
    mas_lambdas = [0.5, 1.0, 0.25]
    # EWC
    parser.add_argument('--EWC_lambda', default=100, type=float, help='The ewc strength')
    # SI
    parser.add_argument('--si_lambda', default=1e1, type=float, help='The SI strength')
    parser.add_argument('--si_eps', default=0.0000001, type=float, help='The SI strength')
    # LFL
    parser.add_argument('--LFL_lambda', default=0.25, type=float, help='The LFL strength')
    # LWF
    parser.add_argument('--LWF_lambda', default=0.1, type=float, help='The LWF strength')
    parser.add_argument('--temperature_LWF', default=4.0, type=float, help='The LWF temperature')
    # MAS
    parser.add_argument('--mas_alpha', default=0.5, type=float, help='The MAS continual importance weight')
    parser.add_argument('--MAS_lambda', default=mas_lambdas[0], type=float, help='The MAS strength')
    # RWALK
    parser.add_argument('--rwalk_lambda', default=lambda_rwalk[-1], type=float, help='The rwalk strength')
    parser.add_argument('--rwalk_alpha', default=0.9, type=float, help='The rwalk continual importance weight')
    parser.add_argument('--rwalk_delta_t', default=10, type=int, help='The rwalk step')
    #
    parser.add_argument('--SI_lambda', default=0.01, type=float, help='The rwalk strength')
    #
    # General arguments.
    parser.add_argument('--train_mb_size', default=10, type=int, help='The avalanche training bs')
    parser.add_argument('--eval_mb_size', default=10, type=int, help='The avalanche evaluation bs')
    parser.add_argument('--train_epochs', default=40, type=int,
                        help='The number of epochs')
    return parser.parse_args()


def update_parser(parser: argparse, attr: str, new_value: any) -> None:
    """
    Update existing parser attribute to new_value.
    Args:
        parser: A parser
        attr: An attr
        new_value: The new value we want to assign

    """
    setattr(parser, attr, new_value)
