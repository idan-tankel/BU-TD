import argparse
from distutils.command.sdist import sdist
import os
import pickle
import sys

from torch.utils.data import DataLoader

from supp.Dataset_and_model_type_specification import DsType, Flag
from supp.datasets import DatasetAllDataSetTypesAll as dataset
from Configs.Config import Config
from typing import Union

# TODO - GET RIT OF THIS.
sys.path.append(r'/home/idanta/BU-TD/yonathan/Recognicion/code/create_dataset')


def get_dataset_for_spatial_realtions(opts: Union[argparse.ArgumentParser, Config], data_fname: str, lang_idx: int, direction: int) -> list:
    """
    Getting the train,test,val(if exists) datasets.
    Args:
        opts: The model options.
        data_fname: The data path.
        lang_idx: The language index.
        direction: The direction.

    Returns: The train, test, val(if exists) datasets.
    (`tuple`): train_dl, test_dl, val_dl, train_ds, test_ds, val_ds

    """
    try:
        model_flag = opts.RunningSpecs.Flag
    except AttributeError:
        model_flag = opts.Flag
    if model_flag not in [Flag.NOFLAG,Flag.Attention,Flag.AttentionHuggingFace]:
        from supp.datasets import DatasetAllDataSetTypes as dataset
    else:
        from supp.datasets import DatasetAllDataSetTypesAll as dataset

    path_fname = os.path.join(data_fname, 'MetaData')
    # Opening the conf file and retrieve number of samples, img shape,number of objects per image.
    with open(path_fname, "rb") as new_data_file:
        MetaData = pickle.load(new_data_file)
    image_size = MetaData.parser.image_size
    nsamples_train = MetaData.nsamples_dict['train']
    nsamples_test = MetaData.nsamples_dict['test']
    # if there is no validation set, then the number of samples in the validation set is 0.
    try:
        nsamples_val = MetaData.nsamples_dict['val']
    except KeyError:
        nsamples_val = 0
    obj_per_row = MetaData.parser.nchars_per_row
    obj_per_col = MetaData.parser.num_rows_in_the_image
    # Creating the data-sets.
    # backward compatibilty
    try:
        ds_type = opts.Datasets.dataset
    except AttributeError:
        ds_type = DsType.Emnist
    if ds_type is DsType.Omniglot:
        arg_and_head_index = lang_idx
    else:
        arg_and_head_index = 0
    train_ds = dataset(root=os.path.join(data_fname, 'train'), opts=opts, arg_and_head_index=arg_and_head_index, direction=direction,
                       is_train=True, nexamples=nsamples_train, obj_per_row=obj_per_row, obj_per_col=obj_per_col)
    test_ds = dataset(root=os.path.join(data_fname, 'test'), opts=opts, arg_and_head_index=arg_and_head_index, direction=direction,
                      is_train=False, nexamples=nsamples_test, obj_per_row=obj_per_row, obj_per_col=obj_per_col)
    batch_size = opts.Training.bs

    # Creating the data-loaders.
    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          num_workers=opts.Training.num_workers, shuffle=True, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size,
                         num_workers=opts.Training.num_workers, shuffle=False, pin_memory=True)
    val_dl = None
    #
    nbatches_train = len(train_dl)
    nbatches_test = len(test_dl)
    nbatches_val = 0

    if os.path.exists(os.path.join(data_fname, 'val')):
        val_ds = dataset(os.path.join(data_fname, 'val'), opts,
                         direction, False, nsamples_val, obj_per_row, obj_per_col)
        val_dl = DataLoader(val_ds, batch_size=batch_size,
                            num_workers=opts.Training.num_workers, shuffle=False, pin_memory=True)
        nbatches_val = len(val_dl)
    else:
        val_ds = None
    # Storing tht parameters into the Parser
    opts.img_channels = 3
    opts.image_size = image_size
    opts.inshape = (opts.img_channels, *opts.image_size)
    opts.nbatches_train = nbatches_train
    opts.nbatches_val = nbatches_val
    opts.nbatches_test = nbatches_test
    return train_dl, test_dl, val_dl, train_ds, test_ds, val_ds
