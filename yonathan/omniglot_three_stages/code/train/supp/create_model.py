import os.path
from supp.general_functions import *
from supp.FlagAt import *
from supp.models import *
from supp.training_functions import *
from supp.loss_and_accuracy import UnifiedLossFun

def create_model(opts: argparse) -> nn.Module:
    """
    Creating a model and moving it to the cuda.
    Args:
        opts: The model options we create the model according to.

    Returns: The created model.

    """
    if opts.model_flag is FlagAt.BU1_SIMPLE:
        model = BUModelSimple(opts)
    else:
        model = CYCLICBUTDMODELSHARED(opts)
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