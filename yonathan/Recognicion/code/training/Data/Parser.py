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

    Returns: The options opts.

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
    opts = argparse.ArgumentParser()
    # Flags.
    opts.add_argument('--ds_type', default=ds_type, type=DsType, help='Flag that defines the data-set type')
    opts.add_argument('--model_flag', default=model_flag, type=Flag, help='Flag that defines the model type')
    # Optimization arguments.
    opts.add_argument('--wd', default=0.00001, type=float, help='The weight decay of the Adam optimizer')
    opts.add_argument('--SGD', default=False, type=bool, help='Whether to use SGD or Adam optimizer')
    opts.add_argument('--initial_lr', default=0.0001, type=float, help='Base lr for the SGD optimizer')
    opts.add_argument('--cycle_lr', default=True, type=bool, help='Whether to cycle the lr')
    opts.add_argument('--base_lr', default=0.0002, type=float, help='Base lr of the cyclic scheduler')
    opts.add_argument('--max_lr', default=0.002, type=float,
                      help='Max lr of the cyclic scheduler before the lr returns to the base_lr')
    opts.add_argument('--momentum', default=0.9, type=float, help='Momentum of the optimizer')
    opts.add_argument('--bs', default=10, type=int, help='The training batch size')
    opts.add_argument('--EPOCHS', default=100, type=int, help='Number of epochs in the training')
    # Model architecture arguments.
    opts.add_argument('--model_type', default=model_type, type=DsType, help='The model type BUTD or ResNet')
    opts.add_argument('--td_block_type', default=BasicBlockTD, type=nn.Module, help='Basic TD block')
    opts.add_argument('--bu_block_type', default=BasicBlockBU, type=nn.Module, help='Basic BU block')
    opts.add_argument('--bu_shared_block_type', default=BasicBlockBUShared, type=nn.Module,
                      help='Basic shared BU block')
    opts.add_argument('--activation_fun', default=nn.ReLU, type=nn.Module, help='The non linear activation function')
    opts.add_argument('--norm_layer', default=BatchNorm, type=nn.Module, help='The used batch normalization')
    opts.add_argument('--use_lateral_butd', default=True, type=bool,
                      help='Whether to use the lateral connection from BU1 to TD')
    opts.add_argument('--use_lateral_tdbu', default=True, type=bool,
                      help='Whether to use the lateral connection from TD to BU2')
    opts.add_argument('--nfilters', default=[64, 96, 128, 256], type=list, help='The ResNet filters')
    opts.add_argument('--strides', default=[2, 2, 1, 2], type=list, help='The ResNet strides')
    opts.add_argument('--ks', default=[7, 3, 3, 3], type=list, help='The ResNet kernel sizes')
    opts.add_argument('--ns', default=[1, 1, 1], type=list, help='Number of blocks per layer')
    opts.add_argument('--nclasses', default=Data_specification.data_obj.nclasses, type=list,
                      help='The sizes of the task-head classes')
    opts.add_argument('--ntasks', default=Data_specification.data_obj.ntasks, type=int,
                      help='Number of tasks the model should solve')
    opts.add_argument('--ndirections', default=Data_specification.data_obj.ndirections, type=int,
                      help='Number of directions the model should handle')
    opts.add_argument('--inshape', default=(3, *Data_specification.data_obj.image_size), type=tuple,
                      help='The input image shape, may be override in get_dataset')
    opts.add_argument('--num_heads', default=Data_specification.data_obj.num_heads, type=list,
                      help='The number of headed for each task, direction')
    opts.add_argument('--num_x_axis', default=Data_specification.data_obj.num_x_axis, type=int,
                      help='The neighbor radius in the x-axis.')
    opts.add_argument('--num_y_axis', default=Data_specification.data_obj.num_y_axis, type=int,
                      help='The neighbor radius in the y-axis.')
    opts.add_argument('--shared', default=True, type=bool,
                      help='Whether the conv layers of BU1, BU2 should be identical.')
    # Training and evaluation arguments.
    opts.add_argument('--use_bu1_loss', default=Data_specification.data_obj.use_bu1_loss, type=bool,
                      help='Whether to use the binary classification loss at the end of the BU1 stream')
    opts.add_argument('--bu1_loss', default=nn.BCEWithLogitsLoss(), type=Callable,
                      help='The loss used at the end of the bu1 stream')
    opts.add_argument('--bu2_loss', default=Data_specification.data_obj.bu2_loss, type=Callable,
                      help='The bu2 classification loss')
    opts.add_argument('--criterion', default=UnifiedCriterion, type=Callable,
                      help='The unified loss function of all training')
    opts.add_argument('--task_accuracy', default=Data_specification.data_obj.task_accuracy, type=Callable,
                      help='The Accuracy function')
    # Struct variables.
    opts.add_argument('--inputs_to_struct', default=inputs_to_struct, type=ClassVar,
                      help='The struct transform the list of inputs to struct.')
    opts.add_argument('--outs_to_struct', default=outs_to_struct, type=ClassVar,
                      help='The struct transform the list of outputs to struct.')
    # Data arguments.
    opts.add_argument('--device', default=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                      type=str, help='The device we use,usually cuda')
    opts.add_argument('--workers', default=2, type=int, help='Number of workers to use')
    # Training procedure.
    opts.add_argument('--initial_directions', default=Data_specification.data_obj.initial_directions, type=list,
                      help='The initial tasks to train first')
    # Change to another function
    model_path = "Model_task_{}_direction_{}_ds_{}".format(task_idx, direction_idx, str(ds_type))
    opts.add_argument('--baselines_dir', default=Data_specification.data_obj.baselines_dir, type=str,
                      help='The path the model will be stored')
    opts.add_argument('--results_dir', default=Data_specification.data_obj.results_dir, type=str,
                      help='The path the model will be stored')
    model_dir = os.path.join(Data_specification.data_obj.results_dir, model_path)
    opts.add_argument('--model_dir', default=model_dir, type=str, help='The model path')
    ########################################
    # For Avalanche Baselines only.
    ########################################
    lambda_rwalk = [20.0, 30.0, 40.0, 50.0, 10.0, 5.0, 2.5]
    # EWC
    opts.add_argument('--EWC_lambda', default=1e20, type=float, help='The ewc strength')
    # SI
    opts.add_argument('--si_lambda', default=1e1, type=float, help='The SI strength')
    opts.add_argument('--si_eps', default=0.0000001, type=float, help='The SI strength')
    # LFL
    opts.add_argument('--LFL_lambda', default=0.70, type=float, help='The LFL strength')
    # LWF
    opts.add_argument('--LWF_lambda', default=0.07, type=float, help='The LWF strength')
    opts.add_argument('--temperature_LWF', default=2.0, type=float, help='The LWF temperature')
    # MAS
    opts.add_argument('--mas_alpha', default=0.5, type=float, help='The MAS continual importance weight')
    opts.add_argument('--MAS_lambda', default=16000 * 4.50, type=float, help='The MAS strength')
    # RWALK
    opts.add_argument('--rwalk_lambda', default=lambda_rwalk[-1], type=float, help='The rwalk strength')
    opts.add_argument('--rwalk_alpha', default=0.9, type=float, help='The rwalk continual importance weight')
    opts.add_argument('--rwalk_delta_t', default=10, type=int, help='The rwalk step')
    #

    opts.add_argument('--imm_mean_lambda', default=0.70, type=float, help='The imm_mean strength')
    opts.add_argument('--imm_mode_lambda', default=0.70, type=float, help='The imm_mean strength')

    #
    opts.add_argument('--SI_lambda', default=0.01, type=float, help='The rwalk strength')
    #
    # General arguments.
    opts.add_argument('--train_mb_size', default=10, type=int, help='The avalanche training bs')
    opts.add_argument('--eval_mb_size', default=10, type=int, help='The avalanche evaluation bs')
    opts.add_argument('--train_epochs', default=40, type=int,
                      help='The number of epochs')
    return opts.parse_args()


def update_parser(opts: argparse, attr: str, new_value: any) -> None:
    """
    Update existing model opts attribute to new_value.
    Args:
        opts: A model opts.
        attr: An attr.
        new_value: The new value we want to assign

    """
    setattr(opts, attr, new_value)
