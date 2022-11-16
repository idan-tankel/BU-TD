import os.path
import argparse
from types import SimpleNamespace
from torch import nn as nn
import torch
import logging
import git
from typing import Union
# from v26.models.BU_TD_Models import BUTDModelShared,BUModelSimple
# from v26 import ConstantsBuTd
# from v26.funcs import logger
from supp.Dataset_and_model_type_specification import Flag
# from supp.models import BUTDModelShared,BUTDModel,BUModel,BUStream,BUStreamShared
from models.BU_TD_Models import BUTDModelShared, BUModelSimple
from models.Attention import Attention,Attention2
logger = logging.getLogger(__name__)

try:
    from Configs.Config import Config,Models
except ImportError:
    from code.Configs.Config import Config,Models


def get_or_create_model(model_opts:Config) -> nn.Module:
    """
    If the model exists, and specified in the config, it will be loaded.
    Creating a model and make it parallel and move it to the cuda.
    :param args: arguments to create the model according to.
    :return: The desired model.
    """
    if model_opts.Training.load_existing_path:
        model = nn.Module() # creare an e model
        assert os.path.exists(model_opts.Training.path_loading), f"Path {model_opts.Training.path_loading} does not exist."
        model.load_state_dict(torch.load(model_opts.Training.path_loading))
    switcher = {
        Flag.TD: BUTDModelShared,
        Flag.NOFLAG: BUModelSimple,
        Flag.ZF: BUTDModelShared,
        Flag.BU2: BUTDModelShared,
        Flag.BU1: BUTDModelShared,
        Flag.BU1_SIMPLE: BUModelSimple,
        Flag.BU1_NOFLAG: BUModelSimple,
        Flag.SF: BUTDModelShared,
        Flag.AttentionHuggingFace: Attention2,
        Flag.Attention: Attention
    }
    try:
        model = switcher[model_opts.RunningSpecs.Flag](model_opts)
    except KeyError:
        model_opts = model_opts.Models
        model = switcher[model_opts.RunningSpecs.Flag](model_opts)
        # try to fix local problems with the transition from config file to a parser
    # model = nn.DataParallel(model)
    model = model.cuda()
    return model