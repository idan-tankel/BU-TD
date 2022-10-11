import logging
import os
import shutil
import sys
import torch
import setuptools
import argparse


def reset_logger(logger: logging) -> None:
    """
    Resting the logger.
    Args:
        logger: The logger.

    Returns:

    """
    while len(logger.handlers) > 0:
        logger.handlers.pop()


def setup_logger(logger: logging, fname: str) -> logging:
    """
    Args:
        logger: The logger.
        fname: Path to the logger.

    Returns: The setup logger.

    """
    reset_logger(logger)
    logger_handler = logging.FileHandler(fname)
    logger_handler.setLevel(logging.DEBUG)
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    return logger


def log_init(opts: argparse) -> None:
    """
    Args:
        opts: The model options.

    """
    logging.basicConfig(format=('%(asctime)s ' + '%(message)s'), level=logging.INFO)
    logger = logging.getLogger(__name__)
    logfname = opts.logfname
    opts.logger = logger
    if logfname is not None:
        model_dir = opts.model_dir
        setup_logger(logger, os.path.join(model_dir, logfname))


def print_info(opts: argparse) -> None:
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


#  opts.logger.info('model_opts: %s', str(opts))

def save_script(opts: argparse) -> None:
    """
    Saving the code script.
    Args:
        opts: Model options.

    """
    model_dir = opts.model_dir
    if getattr(opts, 'save_script', None) is None:
        save_script = True
    else:
        save_script = opts.save_script

    if save_script:
        import __main__ as main
        # copy the running script
        script_fname = main.__file__
        shutil.copy(script_fname, model_dir)
        
        mods = 'supp'
        if len(mods) > 0:
            mods = mods
            funcs_version = mods
            if not os.path.exists(os.path.join(model_dir, funcs_version)):
                shutil.copytree(funcs_version, os.path.join(model_dir, funcs_version))


def print_detail(args: argparse) -> None:
    """
    Args:
        args: The model options.

    """
    first_node = True
    args.first_node = first_node
    if first_node:
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
    log_init(args)
    if first_node:
        print_info(args)
        save_script(args)
    log_msgs = []
    for msg in log_msgs:
        logger.info(msg)