import os.path
import argparse
from types import SimpleNamespace
from torch import nn as nn
import torch
import logging
# from general_functions import *
# from Flag import *
# from models import *
# from training_functions import *
# from loss_and_accuracy import UnifiedLossFun
from typing import Union
from Configs.Config import Config
# from v26.models.BU_TD_Models import BUTDModelShared,BUModelSimple
# from v26 import ConstantsBuTd
# from v26.funcs import logger
from .Dataset_and_model_type_specification import Flag
# from supp.models import BUTDModelShared,BUTDModel,BUModel,BUStream,BUStreamShared
from .models import BUTDModelShared, BUModelSimple
logger = logging.getLogger(__name__)


def create_model(model_opts: Config) -> nn.Module:
    """
    Creating a model and make it parallel and move it to the cuda.
    :param args: arguments to create the model according to.
    :return: The desired model.
    """
    if model_opts.RunningSpecs.Flag is Flag.BU1_SIMPLE:
        model = BUModelSimple(model_opts)
    else:
        model = BUTDModelShared(model_opts)
    if not torch.cuda.is_available():
        logger.info('using CPU, this will be slow')
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
    # ConstantsBuTd.set_model(model)
    # ConstantsBuTd.set_model_opts(model_opts)

    if model_opts.RunningSpecs.Flag is Flag.BU1_SIMPLE:
        model = BUModelSimple(opts=model_opts)
    else:
        model = BUTDModelShared(config=model_opts)

    return model
    # since the pytorch lightning trainer is used, the model is created here and the dev is controlled by the trainer
