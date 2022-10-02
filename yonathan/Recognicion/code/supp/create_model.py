import argparse
from supp.FlagAt import DsType
import torch
import torch.nn as nn
from supp.models import ResNet, BUTDModelShared


def create_model(opts: argparse) -> nn.Module:
    """
    Creating the model according to the opts, the default is BUTD model.
    Args:
        opts: The model options.

    Returns:

    """
    if opts.ds_type is DsType.Cifar10:
        model = ResNet(opts)
    if opts.ds_type is DsType.Emnist or opts.ds_type is DsType.Omniglot or opts.ds_type is DsType.FashionMnist:
        model = BUTDModelShared(opts)
    if not torch.cuda.is_available():
        logger.info('using CPU, this will be slow')
    elif opts.distributed:
        if opts.gpu is not None:
            torch.cuda.set_device(opts.gpu)
            model.cuda(opts.gpu)
            opts.workers = int((opts.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opts.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif opts.gpu is not None:
        torch.cuda.set_device(opts.gpu)
        model = model.cuda(opts.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    return model
