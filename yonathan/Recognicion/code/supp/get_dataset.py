import argparse
import os
import pickle
import sys

from torch.utils.data import DataLoader

from supp.Dataset_and_model_type_specification import DsType, Flag
from supp.datasets import DatasetAllDataSetTypesAll as dataset


# TODO - GET RIT OF THIS.
sys.path.append(r'/home/idanta/BU-TD/yonathan/Recognicion/code/create_dataset')


def get_dataset_for_spatial_realtions(opts: argparse, data_fname: str, lang_idx: int, direction: int) -> list:
    """
    Getting the train,test,val(if exists) datasets.
    Args:
        opts: The model options.
        data_fname: The data path.
        lang_idx: The language index.
        direction: The direction.

    Returns: The train, test, val(if exists) datasets.

    """

    if opts.model_flag is not Flag.NOFLAG:
        from supp.datasets import DatasetAllDataSetTypes as dataset
    elif opts.model_flag is Flag.NOFLAG:
        from supp.datasets import DatasetAllDataSetTypesAll as dataset

    path_fname = os.path.join(data_fname, 'MetaData')
    # Opening the conf file and retrieve number of samples, img shape,number of objects per image.
    with open(path_fname, "rb") as new_data_file:
        MetaData = pickle.load(new_data_file)
    image_size = MetaData.parser.image_size
    nsamples_train = MetaData.nsamples_dict['train']
    nsamples_test = MetaData.nsamples_dict['test']
    nsamples_val = MetaData.nsamples_dict['val']
    obj_per_row = MetaData.parser.nchars_per_row
    obj_per_col = MetaData.parser.num_rows_in_the_image
    # Creating the data-sets.
    if opts.ds_type is DsType.Omniglot:
        arg_and_head_index = lang_idx
    else:
        arg_and_head_index = 0
    train_ds = dataset(root=os.path.join(data_fname, 'train'), opts=opts, arg_and_head_index = arg_and_head_index, direction=direction,
                       is_train=True, nexamples=nsamples_train, obj_per_row=obj_per_row, obj_per_col=obj_per_col)
    test_ds = dataset(root=os.path.join(data_fname, 'test'), opts=opts, arg_and_head_index=arg_and_head_index, direction=direction,
                      is_train=False, nexamples=nsamples_test, obj_per_row=obj_per_row, obj_per_col=obj_per_col)
    batch_size = opts.bs

    # Creating the data-loaders.
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=opts.workers, shuffle=True, pin_memory=True )
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=opts.workers, shuffle=False, pin_memory=True)
    val_dl = None
    #
    nbatches_train = len(train_dl)
    nbatches_test = len(test_dl)
    nbatches_val = 0

    if os.path.exists(os.path.join(data_fname, 'val')):
        val_ds = dataset(os.path.join(data_fname, 'val'), opts, direction,False, nsamples_val, obj_per_row, obj_per_col)
        val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=opts.workers, shuffle=False, pin_memory=True)
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
    return [train_dl, test_dl, val_dl, train_ds, test_ds, val_ds]
