from types import SimpleNamespace

import numpy as np
from torch import nn

from v26.Configs.Config import Config
from v26.funcs import setup_flag
from v26.models.flag_at import FlagAt
from vae.StoreForVae import StoreForVae


def add_arguments_to_parser(parser):
    parser.add_argument('--workers',
                        default=8,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--hyper', default=-1, type=int)
    parser.add_argument('--only_cont', action='store_true')
    parser.add_argument('--SGD', action='store_true')
    parser.add_argument('-lr', default=1e-3 / 2, type=float)
    parser.add_argument('-bs', default=10, type=int)
    parser.add_argument('-wd', default=0.0001, type=float)
    parser.add_argument('--checkpoints-per-epoch', default=1, type=int)
    parser.add_argument('-e',
                        '--extended',
                        action='store_true',
                        help='Use the extended set instead of the sufficient set')


def get_num_features(num_heads: str, ntypes, num_features: int):
    if num_heads == "only_features":
        return num_features
    return len(ntypes)


def init_model_options(config: Config, flag_at, normalize_image, nclasses_existence: int, ntypes: int, flag_size,
                       BatchNorm, inshape):
    """
    init_model_options initialize the core model options

    Args:
        config (Config): The config object represent the config file
        flag_at (_type_): _description_
        normalize_image (_type_): _description_
        nclasses_existence (int): _description_
        ntypes (int): _description_
        flag_size (_type_): _description_
        BatchNorm (_type_): _description_
        inshape (_type_): _description_

    Returns:
        `SimpleNamespae`: the core model options from the config file
    """                       
    model_opts = SimpleNamespace()
    model_opts.data_dir = config.Folders.data_dir
    model_opts.normalize_image = normalize_image
    model_opts.flag_at = flag_at
    model_opts.nclasses_existence = nclasses_existence
    model_opts.head_of_all_features = get_num_features(config.Models.num_heads, ntypes, 7)  # nfeatures=7

    if model_opts.flag_at is FlagAt.NOFLAG:
        ntypes = ntypes[:model_opts.head_of_all_features]
        model_opts.nclasses = ntypes  # with this we get matrix of losses
    else:
        model_opts.nclasses = [ntypes[0]]

    model_opts.flag_size = flag_size
    model_opts.norm_fun = BatchNorm
    model_opts.activation_fun = nn.ReLU
    model_opts.use_td_loss = config.Losses.use_td_loss
    model_opts.use_bu1_loss = config.Losses.use_bu1_loss
    model_opts.use_bu2_loss = config.Losses.use_bu2_loss
    model_opts.use_lateral_butd = True
    model_opts.use_lateral_tdbu = True
    model_opts.use_final_conv = False
    model_opts.ntaskhead_fc = 1
    setup_flag(model_opts)

    # based on ResNet 18
    model_opts.nfilters = [64, 64, 128, 256, 512]
    model_opts.strides = [2, 2, 2, 2, 2]
    # filter sizes
    model_opts.ks = [7, 3, 3, 3, 3]
    model_opts.ns = [0, 2, 2, 2, 2]
    model_opts.inshape = inshape
    if model_opts.flag_at is FlagAt.BU1_SIMPLE:
        model_opts.ns = 3 * np.array(model_opts.ns)  # results in 56 layers
    return model_opts
