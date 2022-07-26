import os.path
from supp.general_functions import *
from supp.FlagAt import *
from supp.models import *
from supp.training_functions import *
from supp.loss_and_accuracy import UnifiedLossFun


def create_model(args: argparse) -> nn.Module:
    """
    Creating a model and make it parallel and move it to the cuda.
    :param args: arguments to create the model according to.
    :return: The desired model.
    """
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
