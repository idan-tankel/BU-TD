import argparse
import os
import pickle
import sys
from pathlib import Path

from torch.utils.data import DataLoader

from training.Data.Data_params import Flag, DsType

sys.path.append(os.path.join(Path(__file__).parents[2], 'data'))


def get_dataset_for_spatial_relations(opts: argparse, data_fname: str, lang_idx: int,
                                      direction: tuple[int, int]) -> dict:
    """
    Getting the train,test,val(if exists) datasets.
    Args:
        opts: The model options.
        data_fname: The data path.
        lang_idx: The language index.
        direction: The direction tuple.

    Returns: The train_dl, test_dl, val_dl(if exists) datasets.
    """
    Data_loader_dict = {}

    if opts.model_flag is not Flag.NOFLAG:  # Import 'All' dataset.
        from training.Data.Datasets import DatasetGuided as dataset
    else:  # Import 'task specific' dataset.
        from training.Data.Datasets import DatasetNonGuided as dataset

    path_fname = os.path.join(data_fname, 'MetaData')
    # Opening the conf file and retrieve number of samples, img shape, number of objects per image.
    with open(path_fname, "rb") as new_data_file:
        MetaData = pickle.load(new_data_file)
    batch_size = opts.bs
    image_size = MetaData.parser.image_size  # Get the image size.
    opts.inshape = (3, *image_size[1:])  # Updating the image size according to the actual data.
    obj_per_row = MetaData.parser.nchars_per_row  # Getting the number of chars per row.
    obj_per_col = MetaData.parser.num_rows_in_the_image  # Getting the number of chars per col.
    nsamples_train = MetaData.nsamples_dict['train']  # Getting the number of train samples.
    nsamples_test = MetaData.nsamples_dict['test']  # Getting the number of test  samples.
    # if there is no validation set, then the number of samples in the validation set is 0.
    try:
        nsamples_val = MetaData.nsamples_dict['val']  # Getting the number of test  samples.
    except KeyError:
        nsamples_val = 0
    # The task index meaningful only for Omniglot.
    task_idx = lang_idx if opts.ds_type is DsType.Omniglot else 0
    # Create train, test dataset according to the opts.
    train_ds = dataset(root=os.path.join(data_fname, 'train'), opts=opts, task_idx=task_idx,
                       direction=direction,
                       is_train=True, nexamples=nsamples_train, obj_per_row=obj_per_row, obj_per_col=obj_per_col)
    test_ds = dataset(root=os.path.join(data_fname, 'test'), opts=opts, task_idx=task_idx,
                      direction=direction,
                      is_train=False, nexamples=nsamples_test, obj_per_row=obj_per_row, obj_per_col=obj_per_col)

    # Creating the data-loaders.
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=opts.workers, shuffle=True,
                          pin_memory=True)  # The Train Data-Loader.
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=opts.workers, shuffle=False,
                         pin_memory=True)  # The Test Data-Loader.
    val_dl = None
    val_ds = None
    #
    if nsamples_val > 0:  # Initialize the val dataset if nsamples_val is non-zero.
        val_ds = dataset(os.path.join(data_fname, 'val'), opts, task_idx, direction, False, nsamples_val,
                         obj_per_row, obj_per_col)
        val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=opts.workers, shuffle=False, pin_memory=True)

    Data_loader_dict['train_ds'] = train_ds
    Data_loader_dict['test_ds'] = test_ds
    Data_loader_dict['val_ds'] = val_ds
    Data_loader_dict['train_dl'] = train_dl
    Data_loader_dict['test_dl'] = test_dl
    Data_loader_dict['val_dl'] = val_dl

    return Data_loader_dict
