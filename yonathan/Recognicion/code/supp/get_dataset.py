import argparse
import os
import pickle
import sys

from torch.utils.data import DataLoader

from supp.Dataset_and_model_type_specification import DsType, Flag
from supp.data_functions import WrappedDataLoader, preprocess
from supp.training_functions import DatasetInfo

sys.path.append(r'/home/sverkip/data/BU-TD/yonathan/Recognicion/code/create_dataset')


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
    if opts.ds_type is DsType.Omniglot and opts.model_flag is Flag.ZF:
        from supp.datasets import OmniglotDataset as dataset
    if (opts.ds_type is DsType.Emnist or opts.ds_type is DsType.FashionMnist) and opts.model_flag is not Flag.NOFLAG:
        from supp.datasets import EmnistDataSet as dataset
    if (opts.ds_type is DsType.Emnist or opts.ds_type is DsType.FashionMnist) and opts.model_flag is Flag.NOFLAG:
        from supp.datasets import EmnistDataSetAll as dataset

    use_val = opts.generalize
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
        train_ds = dataset(os.path.join(data_fname, 'train'), opts, lang_idx, direction, nsamples_train, obj_per_row,
                           obj_per_col)
        test_ds = dataset(os.path.join(data_fname, 'test'), opts, lang_idx, direction, nsamples_test, obj_per_row,
                          obj_per_col)
    else:
        # TODO - OMIT KEYBOARD DEFAULTS.
        train_ds = dataset(os.path.join(data_fname, 'train'), opts, direction, True, nsamples_train, obj_per_row, obj_per_col)
        test_ds = dataset(os.path.join(data_fname, 'test'), opts, direction, False, nsamples_test, obj_per_row, obj_per_col)
    # If normalize_image is True the mean of the dataset is subtracted from every image.
    batch_size = opts.bs

    # Creating the data-loaders.
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=opts.workers, shuffle=True, pin_memory=True )
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=opts.workers, shuffle=False, pin_memory=True)
    val_dl = None
    #
    nbatches_train = len(train_dl)
    nbatches_test = len(test_dl)
    nbatches_val = 0
    train_dataset = WrappedDataLoader(train_dl, preprocess)
    test_dataset = WrappedDataLoader(test_dl, preprocess)
    the_train_dataset = DatasetInfo(True, train_dataset, nbatches_train, 'Train', opts.checkpoints_per_epoch)
    the_test_dataset = DatasetInfo(False, test_dataset, nbatches_test, 'Test', 1)
    the_datasets = [the_train_dataset, the_test_dataset]
    #
    if use_val:
        val_ds = dataset(os.path.join(data_fname, 'val'), opts, direction,False, nsamples_val, obj_per_row, obj_per_col)
        val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=opts.workers, shuffle=False, pin_memory=True)
        nbatches_val = len(val_dl)
        val_dataset = WrappedDataLoader(val_dl, preprocess)
        the_val_dataset = DatasetInfo(False, val_dataset, nbatches_val, 'Val', 1)
        the_datasets.append(the_val_dataset)
    else:
        val_ds = None
    # Storing tht parameters into the Parser
    opts.img_channels = 3
    opts.image_size = image_size
    opts.inshape = (opts.img_channels, *opts.image_size)
    opts.nbatches_train = nbatches_train
    opts.nbatches_val = nbatches_val
    opts.nbatches_test = nbatches_test
    return [the_datasets, train_dl, test_dl, val_dl, train_ds, test_ds, val_ds]
