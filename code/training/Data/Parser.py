"""
Here we define the modl opts needed for all project.
"""
import argparse
from pathlib import Path
from typing import Callable, Union
from typing import Type

import torch
import torch.nn as nn

from Data_Creation.src.Create_dataset_classes import DsType  # Import the Data_Creation set types.
from .Data_params import Flag, DataSetTypeToParams, GenericDataParams
from ..Metrics.Loss import UnifiedCriterion
from ..Modules.Batch_norm import BatchNorm
from ..Modules.Blocks import BasicBlockTD, BasicBlockBU, BasicBlockBUShared
from ..Modules.Models import BUTDModel, ResNet
import torch.optim as optim


def GetParser(task_idx: int = 0, model_type: Type[Union[BUTDModel, ResNet]] = BUTDModel,
              model_flag: Flag = Flag.CL, ds_type: DsType = DsType.Emnist):
    """
    Args:
        task_idx: The task index.
        model_type: The model type  BUTD or ResNet.
        model_flag: The model flag.
        ds_type: The data type e.g. mnist, fashionmnist, omniglot.

    Returns: The options opts.

    """
    # Asserting NOFLAG is used only with ResNet.
    if (model_flag is not Flag.NOFLAG) and model_type is ResNet:
        raise Exception("Pure ResNet can be trained only in NO-FLAG or Read-argument mode.")
    # Asserting the task idx is meaningful only for Omniglot.
    if task_idx != 0 and (ds_type is DsType.Emnist or ds_type is DsType.Fashionmnist):
        raise Exception("The task id is used only for Omniglot.")
    #
    Data_specification = DataSetTypeToParams(ds_type=ds_type, flag_at=model_flag,
                                             initial_task_for_omniglot_only=5)
    opts = argparse.ArgumentParser()
    # Flags.
    opts.add_argument('--data_obj', default=Data_specification, type=Type[GenericDataParams])
    opts.add_argument('--ds_type', default=ds_type, type=DsType, help='Flag that defines the data-set type.')
    opts.add_argument('--model_flag', default=model_flag, type=Flag, help='Flag that defines the model type.')
    # Optimization arguments.
    opts.add_argument('--wd', default=1e-5, type=float, help='The weight decay of the Adam optimizer.')
    opts.add_argument('--scheduler_type', default=None, type=bool,
                      help='Whether to cycle the lr.')
    opts.add_argument('--threshold', default=5e-3, type=float, help='The weight decay of the Adam optimizer.')
    opts.add_argument('--gamma', default=0.5, type=float, help='The weight decay of the Adam optimizer.')
    opts.add_argument('--initial_lr', default=1e-3, type=float, help='Base lr of the cyclic scheduler.')
    opts.add_argument('--weight_modulation', default=True, type=bool, help='')
    opts.add_argument('--weight_modulation_factor', default=[4, 4, 1, 1], type=list, help='')
    opts.add_argument('--momentum', default=0.9, type=float, help='Momentum of the optimizer.')
    opts.add_argument('--bs', default=64, type=int, help='The training batch size.')
    # Model architecture arguments.
    opts.add_argument('--model_type', default=model_type, type=DsType, help='The model type BUTD or ResNet.')
    opts.add_argument('--td_block_type', default=BasicBlockTD, type=nn.Module, help='Basic TD block.')
    opts.add_argument('--bu_block_type', default=BasicBlockBU, type=nn.Module, help='Basic BU block.')
    opts.add_argument('--bu_shared_block_type', default=BasicBlockBUShared, type=nn.Module,
                      help='Basic shared BU block')
    opts.add_argument('--use_lateral_butd', default=True, type=bool,
                      help='Whether to use the lateral connection from BU1 to TD')
    opts.add_argument('--use_lateral_tdbu', default=True, type=bool,
                      help='Whether to use the lateral connection from TD to BU2')
    opts.add_argument('--nfilters', default=[64, 64, 96, 96], type=list, help='The ResNet filters')
    opts.add_argument('--strides', default=[2, 2, 1, 2], type=list, help='The ResNet strides')
    opts.add_argument('--ks', default=[7, 3, 3, 3], type=list, help='The ResNet kernel sizes')
    opts.add_argument('--pad', default=[3, 1, 1, 1])
    opts.add_argument('--num_blocks', default=[1, 1, 1], type=list, help='Number of blocks per layer')
    opts.add_argument('--shared', default=True, type=bool,
                      help='Whether the conv layers of BU1, BU2 should be identical.')
    # Training and evaluation arguments.
    opts.add_argument('--bu1_loss', default=nn.BCEWithLogitsLoss(), type=Callable,
                      help='The loss used at the end of the bu1 stream')
    opts.add_argument('--criterion', default=UnifiedCriterion, type=Callable,
                      help='The unified loss function of all training')
    # Data arguments.
    opts.add_argument('--device', default=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                      type=str, help='The device we use,usually cuda')
    opts.add_argument('--workers', default=2, type=int, help='Number of workers to use')
    # Training procedure.
    opts.add_argument('--project_path', default=Path(__file__).parents[3], type=str, help='The project path')
    ########################################
    # For Avalanche Baselines only.
    ########################################
    # EWC
    opts.add_argument('--EWC_lambda', default=0.985, type=float, help='The ewc strength')
    # LFL
    opts.add_argument('--LFL_lambda', default=0.5, type=float, help='The LFL strength')
    # LWF
    opts.add_argument('--LWF_lambda', default=0.985, type=float, help='The LWF strength')
    opts.add_argument('--temperature_LWF', default=2.0, type=float, help='The LWF temperature')
    # MAS
    opts.add_argument('--mas_alpha', default=0.5, type=float, help='The MAS continual importance weight')
    opts.add_argument('--MAS_lambda', default=0.001, type=float, help='The MAS strength')
    # RWALK
    opts.add_argument('--rwalk_lambda', default=0.5, type=float, help='The rwalk strength')
    opts.add_argument('--rwalk_alpha', default=0.9, type=float, help='The rwalk continual importance weight')
    opts.add_argument('--rwalk_delta_t', default=10, type=int, help='The rwalk step')
    # IMM
    opts.add_argument('--IMM_Mean_lambda', default=0.01, type=float, help='The imm_mean strength')
    opts.add_argument('--IMM_Mode_lambda', default=0.70, type=float, help='The imm_mean strength')
    # Naive
    opts.add_argument('--Naive_lambda', const=0, action='store_const', help='The Naive strength')
    # SI
    opts.add_argument('--SI_eps', default=0.0000001, type=float, help='The SI strength')
    opts.add_argument('--SI_lambda', default=0.005, type=float, help='The rwalk strength')
    # General baseline arguments.
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
