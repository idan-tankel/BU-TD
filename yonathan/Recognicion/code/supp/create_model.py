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
from models.Attention import Attention
logger = logging.getLogger(__name__)

try:
    from Configs.Config import Config,Models
except ImportError:
    from code.Configs.Config import Config,Models


def create_model(model_opts:Config) -> nn.Module:
    """
    Creating a model and make it parallel and move it to the cuda.
    :param args: arguments to create the model according to.
    :return: The desired model.
    """
    switcher = {
        Flag.TD: BUTDModelShared,
        Flag.NOFLAG: BUModelSimple,
        Flag.ZF: BUTDModelShared,
        Flag.BU2: BUTDModelShared,
        Flag.BU1: BUTDModelShared,
        Flag.BU1_SIMPLE: BUModelSimple,
        Flag.BU1_NOFLAG: BUModelSimple,
        Flag.SF: BUTDModelShared,
        Flag.AttentionPretrained: BUTDModelShared,
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
        model = BUTDModelShared(opts=model_opts)

    return model
    # since the pytorch lightning trainer is used, the model is created here and the dev is controlled by the trainer
    if not torch.cuda.is_available():
        logger.info('using CPU, this will be slow')
    elif args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.workers = int(
                (args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()
