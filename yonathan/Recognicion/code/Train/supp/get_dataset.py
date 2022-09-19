import os
import sys
import pickle
import torch
import argparse
from torch.utils.data import DataLoader
from supp.FlagAt import DsType
from supp.training_functions import DatasetInfo
from supp.data_functions import WrappedDataLoader, preprocess
sys.path.append(r'/home/sverkip/data/BU-TD/yonathan/Recognicion/code/create_dataset')
from Create_dataset_classes import MetaData

def get_dataset_for_spatial_realtions(opts, data_path, lan_idx: int, direction_idx: int,arg_idx:int) -> list:
    """
    Args:
        opts: The model options.
        data_path: The path.
        lan_idx: The language index.
        direction_idx: The direction index.

    Returns: The data sets.

    """
    if opts.ds_type is DsType.Omniglot:
     from supp.datasets import OmniglotDataset as dataset
    else:
     from supp.datasets import EmnistDataSet as dataset

    path_fname = os.path.join(data_path, 'MetaData')
    # Opening the conf file and retrieve number of samples, img shape,number of objects per image.
    with open(path_fname, "rb") as new_data_file:
       from_dump_MetaData = pickle.load(new_data_file)
    old_data_format = True
    if old_data_format:
        image_size = from_dump_MetaData.parser.image_size
        nsamples_train = from_dump_MetaData.nsamples_dict['train']
        nsamples_test = from_dump_MetaData.nsamples_dict['test']
        nsamples_val = from_dump_MetaData.nsamples_dict['val']
        opts.generelize = opts.generelize and nsamples_val > 0
    else:
        image_size = from_dump_MetaData.parser.image_size
        nsamples_train = from_dump_MetaData.nsamples_train
        nsamples_test = from_dump_MetaData.nsamples_test
        nsamples_val = from_dump_MetaData.nsamples_val
        opts.generelize = opts.generelize and nsamples_val > 0

    # If number of gpus>1 creating larger batch size.
    ubs = opts.ubs  # unified batch scale
    if opts.num_gpus > 1:
        ubs = ubs * opts.num_gpus
  #  inshape = (3, *image_size)  # Inshape for the dataset
    # Creating the dataset
    if opts.ds_type is DsType.Omniglot:
          train_ds = dataset(os.path.join(data_path, 'train'), opts.nclasses, opts.ntasks, opts.ndirections, embedding_idx = lan_idx,direction = direction_idx,nargs =  opts.nargs, nexamples=nsamples_train, split=True, arg_idx = arg_idx)
          test_ds = dataset(os.path.join(data_path, 'test'), opts.nclasses, opts.ntasks, opts.ndirections,embedding_idx =  lan_idx, direction = direction_idx,nargs =  opts.nargs, nexamples=nsamples_test, split=True,arg_idx = arg_idx)
    else:
        train_ds = dataset(os.path.join(data_path, 'train'), opts.nclasses, opts.ntasks,embedding_idx = lan_idx,direction=  direction_idx, nargs=opts.nargs, nexamples=nsamples_train, split=True)
        test_ds =  dataset(os.path.join(data_path, 'test'), opts.nclasses, opts.ntasks,embedding_idx = lan_idx,direction=  direction_idx, nargs=opts.nargs, nexamples=nsamples_test, split=True)

    # If normalize_image is True the mean of the dataset is subtracted from every image.
    batch_size = opts.bs
    if opts.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
        test_sampler = None
    else:
        train_sampler = None
        test_sampler = None
        val_sampler = None
        batch_size = opts.bs * ubs


    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=opts.workers, shuffle=True, pin_memory=True, sampler=train_sampler)
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=opts.workers, shuffle=False, pin_memory=True, sampler=test_sampler)

    nbatches_train = len(train_dl)
    nbatches_test = len(test_dl)

    train_dataset = WrappedDataLoader(train_dl, preprocess)
    test_dataset = WrappedDataLoader(test_dl, preprocess)

    the_train_dataset = DatasetInfo(True, train_dataset, nbatches_train, 'Train', opts.checkpoints_per_epoch, train_sampler)
    the_test_dataset = DatasetInfo(False, test_dataset, nbatches_test, 'Test', 1, test_sampler)
    the_datasets = [the_train_dataset, the_test_dataset]

    if opts.generelize:
         val_ds =  dataset(os.path.join(data_path, 'val'), opts.nclasses, opts.ntasks,embedding_idx = lan_idx,direction=  direction_idx, nargs=opts.nargs, nexamples=nsamples_val, split=True)
         val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=opts.workers, shuffle=False, pin_memory=True, sampler=val_sampler)
         nbatches_val = len(val_dl)
         val_dataset = WrappedDataLoader(val_dl, preprocess)
         the_val_dataset = DatasetInfo(False, val_dataset, nbatches_val, 'Val', 1, val_sampler)
         the_datasets.append(the_val_dataset)

    else:
         val_dl = None
         nbatches_val = 0
         val_dataset = None

    # Storing tht parameters into the Parser
    opts.img_channels = 3
    opts.image_size = image_size
    opts.inshape = (opts.img_channels, *opts.image_size)
    opts.ubs = ubs
    opts.nbatches_train = nbatches_train
    opts.nbatches_val = nbatches_val
    opts.nbatches_test = nbatches_test

    return [the_datasets, train_dl, test_dl,val_dl, train_dataset, test_dataset, val_dataset]


def get_dataset_cifar(args: argparse, root: str) -> list:
    from supp.datasets import cifar_dataset as dataset
    """
    returns the datasets needed for the training according to the data_fname,embedding_idx
    :param embedding_idx: the embedding for the task,creates the flag according to the embedding.
    :param args: crated the dataset according to the args
    :param data_fname: root to the files
    :return: List of the datasets,train_dl,test_dl.

    """
    path_fname = os.path.join(root, 'conf')
    # Opening the conf file and retrieve number of samples, img shape,number of objects per image.

    # If normalize_image is True the mean of the dataset is subtracted from every image.
    batch_size = args.bs
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
        test_sampler = None
    else:
        train_sampler = None
        test_sampler = None
        batch_size = args.bs

    train_ds = dataset(root=root, ds_type='train')
    test_ds = dataset(root=root, ds_type='test')

    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=args.workers, shuffle=True, pin_memory=True, sampler=train_sampler)
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=args.workers, shuffle=False, pin_memory=True, sampler=test_sampler)
    nbatches_train = len(train_dl)
    nbatches_test = len(test_dl)
    train_dataset = WrappedDataLoader(train_dl, preprocess)
    test_dataset = WrappedDataLoader(test_dl, preprocess)
    the_train_dataset = DatasetInfo(True, train_dataset, nbatches_train, 'Train', args.checkpoints_per_epoch,     train_sampler)
    the_test_dataset = DatasetInfo(False, test_dataset, nbatches_test, 'Test', 1, test_sampler)
    the_datasets = [the_train_dataset, the_test_dataset]
    # Storing tht parameters into the Parser
    args.img_channels = 3
    args.IMAGE_SIZE = (3, 32, 32)
    args.inshape = (3, 32, 32)
    args.ubs = 1
    args.nbatches_train = nbatches_train
    args.nbatches_test = nbatches_test
    return [the_datasets, train_dl, test_dl, train_dataset, test_dataset]

#TODO - DELETE THIS TWO FUNCTIONS.

'''
def retrieve_mean_image(train_dl, inshape, inputs_to_struct, base_samples_dir, store, stop_after=1000):
    mean_image_fname = os.path.join(base_samples_dir, 'mean_image.pkl')
    if not os.path.exists(mean_image_fname):
        mean_image = get_mean_image(train_dl, inshape, inputs_to_struct, stop_after)
        if store:
            with open(mean_image_fname, "wb") as data_file:
                pickle.dump(mean_image, data_file)
    else:
        with open(mean_image_fname, "rb") as data_file:
            mean_image = pickle.load(data_file)
    return mean_image


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

'''