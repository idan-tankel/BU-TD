"""
Here we return the data-set and data-loader objects.
"""
import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Union, Tuple

from torch.utils.data import DataLoader

from .Data_params import Flag

sys.path.append(os.path.join(Path(__file__).parents[2], 'Data_Creation/src'))


# Return the datasets and dataloaders.

def get_dataset_for_spatial_relations(opts: argparse, data_fname: str, task: Tuple[int, Tuple],
                                      data: Union[None, type] = None) -> dict:
    """
    Getting the train,test,val(if exists) datasets.
    Args:
        opts: The model options.
        data_fname: The data path.
        task: The task tuple.
        data: Possible data-set object.

    Returns: The train_dl, test_dl, val_dl(if exists) datasets.
    """
    Data_loader_dict = {}
    # Import 'Non-guided' dataset.
    if opts.model_flag is Flag.NOFLAG:
        from ..Data.Datasets import DatasetNonGuided as dataset
    # Import 'guided' dataset.
    elif opts.model_flag is Flag.CL or opts.model_flag is Flag.Read_argument:
        from ..Data.Datasets import DatasetGuidedSingleTask as dataset
    else:
        raise Exception("You must implement a dataset object to support that model type.")
    if data is not None:
        dataset = data

    path_fname = os.path.join(data_fname, 'MetaData')
    # Opening the conf file and retrieve number of samples, img shape, number of objects per image.
    with open(path_fname, "rb") as new_data_file:
        MetaData = pickle.load(new_data_file)
    image_size = MetaData.parser.image_size  # Get the image size.
    opts.inshape = (3, *image_size)  # Updating the image size according to the actual data.
    obj_per_row = MetaData.parser.num_cols  # Getting the number of chars per row.
    obj_per_col = MetaData.parser.num_rows  # Getting the number of chars per col.

    nsamples_train = MetaData.nsamples_dict['train']  # Getting the number of train samples.
    nsamples_test = MetaData.nsamples_dict['test']  # Getting the number of test samples.
    try:
        nsamples_val = MetaData.nsamples_dict['val']  # Getting the number of validation(CG test) samples.
    except KeyError:
        nsamples_val = 0

    # if there is no validation set, then the number of samples in the validation set is 0.

    # Create train dataset.
    train_ds = dataset(root=os.path.join(data_fname, 'train'), opts=opts, task_struct=task,
                       nexamples=nsamples_train, obj_per_row=obj_per_row, obj_per_col=obj_per_col)
    # Create the test dataset.
    test_ds = dataset(root=os.path.join(data_fname, 'test'), is_train=False, opts=opts, task_struct=task,
                      nexamples=nsamples_test, obj_per_row=obj_per_row, obj_per_col=obj_per_col)
    batch_size = opts.bs
    # Creating the data-loaders.
    train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, num_workers=opts.workers, shuffle=True,
                          pin_memory=True)  # The Train Data-Loader.
    test_dl = DataLoader(dataset=test_ds, batch_size=batch_size, num_workers=opts.workers, shuffle=False,
                         pin_memory=True)  # The Test Data-Loader.
    val_dl = None
    val_ds = None

    # Create the val dataset if nsamples_val is non-zero.
    if nsamples_val > 0:
        val_ds = dataset(root=os.path.join(data_fname, 'val'), is_train=False, opts=opts,
                         task_struct=task,
                         nexamples=nsamples_val, obj_per_row=obj_per_row, obj_per_col=obj_per_col)
        val_dl = DataLoader(dataset=val_ds, batch_size=batch_size, num_workers=opts.workers, shuffle=False,
                            pin_memory=True)

    # Store all datasets and dataloaders in a dictionary.
    Data_loader_dict['train_ds'] = train_ds
    Data_loader_dict['test_ds'] = test_ds
    Data_loader_dict['val_ds'] = val_ds
    Data_loader_dict['train_dl'] = train_dl
    Data_loader_dict['test_dl'] = test_dl
    Data_loader_dict['val_dl'] = val_dl

    return Data_loader_dict
