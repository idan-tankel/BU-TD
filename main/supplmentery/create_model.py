import os.path
import argparse
from types import SimpleNamespace
from torch import nn as nn
import torch
# from general_functions import *
# from FlagAt import *
# from models import *
# from training_functions import *
# from loss_and_accuracy import UnifiedLossFun
from v26.models.BU_TD_Models import BUTDModelShared,BUModelSimple
from v26 import ConstantsBuTd
from v26.funcs import logger
from supplmentery.FlagAt import FlagAt


def create_model(model_opts: SimpleNamespace) -> nn.Module:
    """
    Creating a model and make it parallel and move it to the cuda.
    :param args: arguments to create the model according to.
    :return: The desired model.
    """
    if model_opts.RunningSpecs.FlagAt is FlagAt.BU1_SIMPLE:
        model = BUModelSimple(model_opts)
    else:

        model = BUTDModelShared(model_opts)
    if not torch.cuda.is_available():
        logger.info('using CPU, this will be slow')
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
    ConstantsBuTd.set_model(model)
    ConstantsBuTd.set_model_opts(model_opts)
    return model

    if args.model_flag is FlagAt.BU1_SIMPLE:
        model = BUModelSimple(args)
    else:
        model = BUTDModelShared(args)
    if not torch.cuda.is_available():
        logger.info('using CPU, this will be slow')
    elif args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    return model
