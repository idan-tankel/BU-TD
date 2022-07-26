import logging
import os
import torch
import shutil
import sys
import torch

def reset_logger(logger):
    while len(logger.handlers) > 0:
        logger.handlers.pop()


def setup_logger(logger, fname):
    reset_logger(logger)
    logger_handler = logging.FileHandler(fname)
    logger_handler.setLevel(logging.DEBUG)
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    return logger


def log_init(opts):
    logging.basicConfig(format=('%(asctime)s ' + '%(message)s'), level=logging.INFO)
    logger = logging.getLogger(__name__)
    logfname = opts.logfname
    opts.logger = logger
    if logfname is not None:
        model_dir = opts.model_dir
        setup_logger(logger, os.path.join(model_dir, logfname))


def print_info(opts):
    import __main__ as main
    try:
        script_fname = main.__file__
        logger.info('Executing file %s' % script_fname)
    except:
        pass
    try:
        import subprocess
        result = subprocess.run(
            ['hostname', ''],
            stdout=subprocess.PIPE,
            shell=True)
        res = result.stdout.decode('utf-8')
        opts.logger.info('Running on host: %s' % res)
    except:
        pass

    opts.logger.info('PyTorch version: %s' % torch.__version__)
    opts.logger.info('cuDNN enabled: %s' % torch.backends.cudnn.enabled)
    opts.logger.info('model_opts: %s', str(opts))


def save_script(opts):
    model_dir = opts.model_dir
    if getattr(opts,'save_script',None) is None:
        save_script=True
    else:
        save_script=opts.save_script

    if save_script:
        import __main__ as main
        # copy the running script
        script_fname = main.__file__
        dst = shutil.copy(script_fname, model_dir)
        if opts.distributed:
            # if distributed then also copy the actual script
            script_base_fname = opts.module + '.py'
            script_base_fname = os.path.join(os.path.dirname(script_fname),script_base_fname)
            dst = shutil.copy(script_base_fname, model_dir)

        # copy funcs folder
        mods = [m.__name__ for m in sys.modules.values() if 'supp' in m.__name__]
        if len(mods)>0:
            mods=mods[0]
            mods = mods.split('.')
            funcs_version=mods[0]
            dst = shutil.copytree(funcs_version, os.path.join(model_dir,funcs_version))


def print_detail(args):
    first_node = not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)
    args.first_node = first_node
    if first_node:
        os.makedirs(args.model_dir)
    if args.distributed:
        import torch.distributed as dist
        dist.barrier()
        model_opts.module = args.module
    log_init(args)
    if first_node:
        print_info(args)
        save_script(args)
    log_msgs = []
    for msg in log_msgs:
        logger.info(msg)
