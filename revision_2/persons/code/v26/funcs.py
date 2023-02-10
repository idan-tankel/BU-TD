import tkinter

import numpy as np
# from line_profiler_pycharm import profile
# from v26.models.FlagAt import FlagAt

from persons.code.v26.Configs.ConfigClass import ConfigClass
from persons.code.v26.ConstantsBuTd import get_dev
from persons.code.v26.models.FlagAt import FlagAt

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=120)
import sys
import os
from pathlib import Path

home = str(Path.home())
sys.path.append(os.path.join(home, "code/other_py"))
from types import SimpleNamespace

import shutil
import six
import pickle
import time
import logging
import matplotlib as mpl

try:
    # only used here for determining matplotlib backend
    import v26.cfg as cfg

    use_gui = cfg.gpu_interactive_queue
except:
    use_gui = False
if use_gui:
    # when running without a graphics server (such as xserver) change to: mpl.use('AGG')
    mpl.use('TkAgg')
    # from mimshow import *
else:
    mpl.use('AGG')
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

import torch
import torch.nn as nn
import torch.optim as optim

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
seed = 0
torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.backends.cudnn.benchmark = False
#     #Deterministic mode can have a performance impact, depending on your model. This means that due to the deterministic nature of the model, the processing speed (i.e. processed batch items per second) can be lower than when the model is non-deterministic.
#     torch.backends.cudnn.deterministic = True
# np.random.seed(seed)  # Numpy module.
# random.seed(seed)  # Python random module.

orig_relus = False  # if True use the same number of ReLUs as in the original tensorflow implementation
orig_bn_eps = False  # if True use the batch norm epsilon as in the original tensorflow implementation
if orig_bn_eps:
    bn_eps = 1e-3
else:
    bn_eps = 1e-5

import logging

logging.basicConfig(format=('%(asctime)s ' + '%(message)s'), level=logging.INFO)

logger = logging.getLogger(__name__)


#######################################
#    General functions
#######################################
def reset_logger():
    # there should be a better way than this...
    while len(logger.handlers) > 0:
        logger.handlers.pop()


def setup_logger(fname):
    reset_logger()
    # Create the Handler for logging data to a file
    logger_handler = logging.FileHandler(fname)
    logger_handler.setLevel(logging.DEBUG)
    # Create a Formatter for formatting the log messages
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    # Add the Formatter to the Handler
    logger_handler.setFormatter(logger_formatter)

    # logger = logging.getLogger(__name__)
    # Add the Handler to the Logger
    logger.addHandler(logger_handler)
    return logger


def log_init(opts):
    logfname = opts.logfname
    if logfname is not None:
        model_dir = opts.model_dir
        setup_logger(os.path.join(model_dir, logfname))


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
        logger.info('Running on host: %s' % res)
    except:
        pass

    logger.info('PyTorch version: %s' % torch.__version__)
    logger.info('cuDNN enabled: %s' % torch.backends.cudnn.enabled)
    logger.info('model_opts: %s', str(opts))


def save_script(opts):
    model_dir = opts.model_dir
    if getattr(opts, 'save_script', None) is None:
        save_script = False
    else:
        save_script = opts.save_script

    if save_script:
        import __main__ as main
        try:
            # copy the running script
            script_fname = main.__file__
            if False:
                # fix some permission problems we have on waic, do not copy mode bits
                dst = model_dir
                if os.path.isdir(dst):
                    dst = os.path.join(dst, os.path.basename(script_fname))
                shutil.copyfile(script_fname, dst)
            else:
                dst = shutil.copy(script_fname, model_dir)
                if opts.distributed:
                    # if distributed then also copy the actual script
                    script_base_fname = opts.module + '.py'
                    script_base_fname = os.path.join(os.path.dirname(script_fname), script_base_fname)
                    dst = shutil.copy(script_base_fname, model_dir)

            # copy funcs folder
            mods = [m.__name__ for m in sys.modules.values() if '.funcs' in m.__name__]
            if len(mods) > 0:
                mods = mods[0]
                mods = mods.split('.')
                funcs_version = mods[0]
                # might want to use  dirs_exist_ok=True for Python>3.6
                dst = shutil.copytree(funcs_version, os.path.join(model_dir, funcs_version))
        except:
            pass


def pause_image(fig=None):
    plt.draw()
    plt.show(block=False)
    if fig == None:
        fig = plt.gcf()

    #    fig.canvas.manager.window.activateWindow()
    #    fig.canvas.manager.window.raise_()
    fig.waitforbuttonpress()


def redraw_fig(fig):
    if fig is None:
        return

    # ask the canvas to re-draw itself the next time it
    # has a chance.
    # For most of the GUI backends this adds an event to the queue
    # of the GUI frameworks event loop.
    fig.canvas.draw_idle()
    try:
        # make sure that the GUI framework has a chance to run its event loop
        # and clear any GUI events.  This needs to be in a try/except block
        # because the default implementation of this method is to raise
        # NotImplementedError
        fig.canvas.flush_events()
    except NotImplementedError:
        pass


# ns* functions are the same as the usual functions for lists but for namespaces
def ns_tocpu(ns):
    dc = ns.__dict__
    for key in dc:
        if dc[key] is not None:
            if isinstance(dc[key], list):
                if isinstance(dc[key][0], list):
                    dc[key] = [[curr_in.cpu() for curr_in in curr] for curr in dc[key]]
                else:
                    dc[key] = [curr.cpu() for curr in dc[key]]
                    # dc[key] = [curr[0].detach() for curr in
                    #            dc[key]]  # TODO - for each in the output of the current example
                # dc[key] = [curr.cpu() for curr in dc[key]]
            else:
                dc[key] = dc[key].cpu()


def ns_tonp(ns):
    ns_tocpu(ns)
    dc = ns.__dict__
    for key in dc:
        if dc[key] is not None:
            if isinstance(dc[key], list):
                if isinstance(dc[key][0], list):
                    dc[key] = [[curr_in.numpy() for curr_in in curr] for curr in dc[key]]
                else:
                    dc[key] = [curr.numpy() for curr in dc[key]]
            else:
                dc[key] = dc[key].numpy()


def ns_detach_tonp(ns):
    dc = ns.__dict__
    for key in dc:
        if dc[key] is not None:
            if isinstance(dc[key], list):
                if isinstance(dc[key][0], list):
                    dc[key] = [[curr_in.detach() for curr_in in curr] for curr in dc[key]]
                else:
                    dc[key] = [curr[0].detach() for curr in
                               dc[key]]  # TODO - for each in the output of the current example
            else:
                dc[key] = dc[key].detach()
    ns_tonp(ns)


def preprocess(inputs):
    inputs = [inp.to(get_dev()) for inp in inputs]
    return inputs


def print_total_parameters(model):
    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("{:,}".format(total_parameters))
    return total_parameters


def from_network(inputs, outs, module, inputs_to_struct, convert_to_np=True):
    if convert_to_np:
        inputs = tonp(inputs)
    else:
        inputs = tocpu(inputs)
    outs = module.outs_to_struct(outs)
    if convert_to_np:
        ns_detach_tonp(outs)
    else:
        ns_tocpu(outs)
    samples = inputs_to_struct(inputs)
    return samples, outs


# def get_mean_image(ds,inshape,inputs_to_struct):
#     mean_image= np.zeros(inshape)
#     for i,inputs in enumerate(ds):
#         inputs=tonp(inputs)
#         samples = inputs_to_struct(inputs)
#         mean_image = (mean_image*i + samples.image)/(i+1)
#         if i == nsamples_train - 1:
#             break
#     # mean_image = mean_image.astype(np.int)
#     return mean_image

# set stop_after to None if you want the accurate mean, otherwise set to the number of examples to process
def get_mean_image(dl, inshape, inputs_to_struct, stop_after=1000):
    mean_image = np.zeros(inshape)
    nimgs = 0
    for inputs in dl:
        inputs = tonp(inputs)
        samples = inputs_to_struct(inputs)
        cur_bs = samples.image.shape[0]
        mean_image = (mean_image * nimgs + samples.image.sum(axis=0)) / (nimgs + cur_bs)
        nimgs += cur_bs
        if stop_after and nimgs > stop_after:
            break
    return mean_image


def argmax_by_thresh(a):
    # returns 0 if the first argument is larger than the second one by at least some threshold
    THRESH = 0.01
    index_array = np.argsort(a)
    max_arg = index_array[-1]
    second_highest_arg = index_array[-2]
    max_val = a[max_arg]
    second_highest_val = a[second_highest_arg]
    if max_val > second_highest_val + THRESH:
        return max_arg
    else:
        return -1


def instruct(struct, key):
    return getattr(struct, key, None) is not None


#######################################
#    Model functions
#######################################


def setup_flag(opts):
    if opts.flag_at is FlagAt.BU2:
        opts.use_bu1_flag = False
        opts.use_td_flag = False
        opts.use_bu2_flag = True
        opts.use_td_loss = False
    elif (opts.flag_at is FlagAt.BU1) or (opts.flag_at is FlagAt.BU1_SIMPLE) or (opts.flag_at is FlagAt.BU1_NOLAT):
        opts.use_bu1_flag = True
        opts.use_td_flag = False
        opts.use_bu2_flag = False
        if (opts.flag_at is FlagAt.BU1_SIMPLE) or (opts.flag_at is FlagAt.BU1_NOLAT):
            opts.use_lateral_butd = False
            opts.use_lateral_tdbu = False
        if opts.flag_at is FlagAt.BU1_SIMPLE:
            opts.use_td_loss = False
    elif opts.flag_at is FlagAt.TD:
        opts.use_bu1_flag = False
        opts.use_td_flag = True
        opts.use_bu2_flag = False
    elif opts.flag_at is FlagAt.NOFLAG:
        opts.use_bu1_flag = False
        opts.use_td_flag = True
        opts.use_bu2_flag = False
        opts.use_td_loss = False


def NoNorm(num_channels, dims=2):
    return nn.Identity()


def GroupNorm(num_groups):
    f = lambda num_channels, dims=2: nn.GroupNorm(num_groups, num_channels)
    return f


# don't use population statistics as we share these batch norm layers across
# the BU pillars, where apparently they receive completely different statistics
# leading to wrong estimations when evaluating
def BatchNormNoStats(num_channels, dims=2):
    if dims == 2:
        norm = nn.BatchNorm2d(num_channels, track_running_stats=False)
    else:
        norm = nn.BatchNorm1d(num_channels, track_running_stats=False)
    return norm


def BatchNorm(num_channels, dims=2):
    if dims == 2:
        norm = nn.BatchNorm2d(num_channels, eps=bn_eps, track_running_stats=True)
    else:
        norm = nn.BatchNorm1d(num_channels, eps=bn_eps, track_running_stats=True)
    return norm


def InstanceNorm(num_channels, dims=2):
    if dims == 2:
        norm = nn.InstanceNorm2d(num_channels, track_running_stats=False)
    else:
        norm = nn.Identity()
    return norm


def LocalResponseNorm():
    size = 2
    return nn.LocalResponseNorm(size)


def get_laterals(laterals, layer_id, block_id):
    if laterals is None:
        return None
    if len(laterals) > layer_id:
        layer_laterals = laterals[layer_id]
        if type(layer_laterals) == list and len(layer_laterals) > block_id:
            return layer_laterals[block_id]
        else:
            return layer_laterals
    return None


def init_module_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


#######################################
#    Train functions
#######################################

@profile
def fit(opts, the_datasets, config: ConfigClass):
    """iterate over the datasets and train (or test) them"""
    # if opts.first_node:
    #     logger.info('train_opts: %s', str(opts))
    optimizer = opts.optimizer
    scheduler = opts.scheduler
    datasets_name = [dataset.name for dataset in the_datasets]

    nb_epochs = opts.EPOCHS

    model_dir = opts.model_dir

    model_ext = '.pt'
    model_basename = 'model'
    model_latest_fname = model_basename + '_latest' + model_ext
    model_latest_fname = os.path.join(model_dir, model_latest_fname)

    if not instruct(opts, 'save_details'):
        save_details = SimpleNamespace()
        # only save by maximum accuracy value
        save_details.optimum = -np.inf
        # save_details.save_cmp_fun = np.argmax
        save_details.save_cmp_fun = argmax_by_thresh
        save_details.epoch_save_idx = -1  # last metric: accuracy
        save_details.dataset_id = 1  # from the test dataset
    else:
        save_details = opts.save_details

    optimum = save_details.optimum

    # restore saved model
    last_epoch, optimum = load_old_model(model_latest_fname, optimum, opts, the_datasets)

    fig = None
    st_epoch = last_epoch + 1
    end_epoch = nb_epochs
    if instruct(opts, 'abort_after_epochs') and opts.abort_after_epochs > 0:
        end_epoch = st_epoch + opts.abort_after_epochs

    for epoch in range(st_epoch, end_epoch):
        if opts.first_node:
            logger.info('Epoch {} learning rate: {}'.format(epoch + 1, optimizer.param_groups[0]['lr']))
        # if opts.distributed:
        #     opts.train_sampler.set_epoch(epoch)

        for the_dataset in the_datasets:
            the_dataset.do_epoch(epoch, opts, end_epoch, config)
        logger.info('Epoch {} done'.format(epoch + 1))
        for the_dataset in the_datasets:
            logger.info(
                'Epoch {}, {} {}'.format(epoch + 1, the_dataset.name, the_dataset.measurements.print_epoch()))

        if epoch < nb_epochs - 1:
            # learning rate scheduler
            scheduler.step()

        # save model
        if opts.first_node:
            # When using distributed data parallel, one optimization is to save the model in only one process, reducing write overhead. This is correct because all processes start from the same parameters and gradients are synchronized in backward passes, and hence optimizers should keep setting parameters to the same values 
            if opts.save_model:
                save_by_dataset = the_datasets[save_details.dataset_id]
                measurements = np.array(save_by_dataset.measurements.results)
                new_optimum = False
                epoch_save_value = measurements[epoch, save_details.epoch_save_idx]
                if save_details.save_cmp_fun(
                        [epoch_save_value, optimum]) == 0:
                    optimum = epoch_save_value
                    new_optimum = True
                    logger.info('New optimum: %f' % optimum)

                metadata = dict()
                metadata['epoch'] = epoch
                for the_dataset in the_datasets:
                    measurements = np.array(the_dataset.measurements.results)
                    metadata[the_dataset.name] = measurements
                metadata['optimum'] = optimum

                model_latest_fname = model_basename + '_latest' + model_ext
                model_latest_fname = os.path.join(model_dir, model_latest_fname)
                save_model_and_md(model_latest_fname, metadata, epoch, opts)
                if new_optimum:
                    model_fname = model_basename + '%d' % (epoch + 1) + model_ext
                    model_fname = os.path.join(model_dir, model_fname)
                    shutil.copyfile(model_latest_fname, model_fname)
                    logger.info('Saved model to %s' % model_fname)
                    # save_model_and_md(model_fname,metadata,epoch,opts)

        # plot
        if opts.first_node:
            if epoch == st_epoch:
                n_measurements = len(the_datasets[0].measurements.metrics)
                fig, subplots_axes = plt.subplots(1, n_measurements, figsize=(5, 3))
                if n_measurements == 1:
                    subplots_axes = [subplots_axes]
            else:
                plt.figure(fig.number)
                for ax in subplots_axes:
                    ax.clear()
            for the_dataset in the_datasets:
                the_dataset.measurements.plot(fig, subplots_axes)
            plt.legend(datasets_name)
            if use_gui:
                plt.show(block=False)
            #        draw()
            redraw_fig(fig)
            fig.savefig(os.path.join(model_dir, 'net-train.png'))
    logger.info('Done fit')


def load_old_model(model_latest_fname, optimum, opts, the_datasets):
    last_epoch = -1
    if opts.load_model_if_exists:
        if os.path.exists(model_latest_fname):
            logger.info('Loading model: %s' % model_latest_fname)
            checkpoint = load_model(opts, model_latest_fname, opts.gpu)
            metadata = checkpoint['metadata']
            for the_dataset in the_datasets:
                the_dataset.measurements.results = metadata[the_dataset.name]
            optimum = metadata['optimum']
            last_epoch = metadata['epoch']
            if opts.distributed:
                # synchronize point so all distributed processes would have the same weights
                import torch.distributed as dist
                dist.barrier()
            logger.info('restored model with optimum %f' % optimum)
            logger.info('continuing from epoch: %d' % (last_epoch + 2))
    return last_epoch, optimum


def activated_tasks(number_tasks, flag):
    loss_weight_by_task = torch.zeros((number_tasks)).to(get_dev())
    for i in list(range(flag.shape[0])):  # add to list!
        index_task = torch.where((flag[i])[:6] == 1)[0].item() * 7 + torch.where((flag[i])[6:] == 1)[
            0].item()
        loss_weight_by_task[index_task] = 1
    return loss_weight_by_task


# def get_predicted_feature_value(tasks_heads, flags, sample_index):
#     for layer_number in range(len(tasks_heads)):
#         all_non_zero = flags[:, 6 + layer_number].nonzero()
#         if all_non_zero.shape[0] > 0 and sample_index in all_non_zero[0].tolist():
#             layer, location_in_layer = layer_number, flags[:, 6 + layer_number].nonzero()[0].tolist().index(sample_index)
#             example_output = tasks_heads[layer][location_in_layer, :]
#             return example_output.argmax()
#     return 8


def get_predicted_feature_value(tasks_heads, flags, sample_index):
    all_values = []
    for layer_number in range(len(tasks_heads)):
        all_non_zero = flags[:, 6 + layer_number].nonzero()
        if all_non_zero.shape[0] > 0 and [sample_index] in all_non_zero.tolist():
            # if all_non_zero.shape[0] > 0 and sample_index in all_non_zero[0].tolist():
            layer, location_in_layer = layer_number, flags[:, 6 + layer_number].nonzero().tolist().index([sample_index])
            example_output = tasks_heads[layer][location_in_layer, :]
            all_values.append(example_output.argmax())
            # return example_output.argmax()
    if len(all_values) == 1:
        return all_values[0]
    if len(all_values) == 0:
        return 8
    return all_values
