import argparse
import os
import pickle
import sys

from torch.utils.data import DataLoader

from supp.Dataset_and_model_type_specification import DsType, Flag

# TODO - GET RIT OF THIS.
from pathlib import Path

sys.path.append(os.path.join(Path(__file__).parents[1], 'create_dataset'))


def get_dataset_for_spatial_realtions(opts: argparse, data_fname: str, lang_idx: int, direction: int) -> tuple[
    DataLoader]:
    """
    Getting the train,test,val(if exists) datasets.
    Args:
        opts: The model options.
        data_fname: The data path.
        lang_idx: The language index.
        direction: The direction tuple.

    Returns: The train_dl, test_dl, val_dl(if exists) datasets.
    """

    if opts.model_flag is not Flag.NOFLAG:  # Import 'All' dataset.
        from supp.datasets import DatasetAllDataSetTypes as dataset
    elif opts.model_flag is Flag.NOFLAG:  # Import 'task specific' dataset.
        from supp.datasets import DatasetAllDataSetTypesAll as dataset

    path_fname = os.path.join(data_fname, 'MetaData')
    # Opening the conf file and retrieve number of samples, img shape, number of objects per image.
    with open(path_fname, "rb") as new_data_file:
        MetaData = pickle.load(new_data_file)
    batch_size = opts.bs
    image_size = MetaData.parser.image_size  # Get the image size.
    opts.inshape = (3, *image_size) # Updating the image size according to the actual data.
    obj_per_row = MetaData.parser.nchars_per_row  # Getting the number of chars per row.
    obj_per_col = MetaData.parser.num_rows_in_the_image  # Getting the number of chars per col.
    nsamples_train = MetaData.nsamples_dict['train']  # Getting the number of train samples.
    nsamples_test = MetaData.nsamples_dict['test']  # Getting the number of test  samples.
    nsamples_val = MetaData.nsamples_dict['val']  # Getting the number of test  samples.
    # if there is no validation set, then the number of samples in the validation set is 0.
    # Creating the data-sets.
    arg_and_head_index = lang_idx if opts.ds_type is DsType.Omniglot else 0
    # Create train, test dataset according to the opts.
    train_ds = dataset(root=os.path.join(data_fname, 'train'), opts=opts, arg_and_head_index=arg_and_head_index,
                       direction=direction,
                       is_train=True, nexamples=nsamples_train, obj_per_row=obj_per_row, obj_per_col=obj_per_col)
    test_ds = dataset(root=os.path.join(data_fname, 'test'), opts=opts, arg_and_head_index=arg_and_head_index,
                      direction=direction,
                      is_train=False, nexamples=nsamples_test, obj_per_row=obj_per_row, obj_per_col=obj_per_col)

    # Creating the data-loaders.
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=opts.workers, shuffle=True, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=opts.workers, shuffle=False, pin_memory=True)
    val_dl = None
    #
    if os.path.exists(os.path.join(data_fname, 'val')):  # Initialize the val dataset if exists.
        val_ds = dataset(os.path.join(data_fname, 'val'), opts, arg_and_head_index, direction, False, nsamples_val,
                         obj_per_row, obj_per_col)
        val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=opts.workers, shuffle=False, pin_memory=True)

    return train_dl, test_dl, val_dl
