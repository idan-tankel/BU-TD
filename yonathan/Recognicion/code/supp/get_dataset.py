import os
import pickle
import torch
import argparse
from torch.utils.data import DataLoader
from supp.FlagAt import *
from supp.training_functions import *
from supp.data_functions import *

def get_dataset_for_spatial_realtions(args,data_fname, embedding_idx: int,direction:int) -> list:
    """
    returns the datasets needed for the training according to the data_fname,embedding_idx
    :param embedding_idx: the embedding for the task,creates the flag according to the embedding.
    :param args: crated the dataset according to the args
    :param data_fname: root to the files
    :return: List of the datasets,train_dl,test_dl.

    """
    if args.ds_type is DsType.Omniglot:
     from supp.datasets import OmniglotDataset as dataset
    else:
     from supp.datasets import EmnistDataSet as dataset
    # TODO - ADOPT FOR EACH DATASET.
    use_val = True
    new_data_format = True
    if not new_data_format:
      path_fname = os.path.join(data_fname, 'conf')
    else:
        path_fname = os.path.join(data_fname, 'MetaData')
    # TODO - MAKE IT INTO A METADATA data.
    # Opening the conf file and retrieve number of samples, img shape,number of objects per image.

    with open(path_fname, "rb") as new_data_file:
        if new_data_format:
         MetaData = pickle.load(new_data_file)
        else:
         nsamples_train, nsamples_test, nsamples_val, nclasses, _ ,image_size, num_rows_in_the_image, obj_per_row, num_chars_per_image,ndirections, valid_classes =  pickle.load(new_data_file)
    if new_data_format:
     image_size = MetaData.image_size
     nsamples_train = MetaData.nsamples_train
     nsamples_test = MetaData.nsamples_test

    # If number of gpus>1 creating larger batch size.
    ubs = args.ubs  # unified batch scale
    if args.num_gpus > 1:
        ubs = ubs * num_gpus
    inshape = (3, *image_size)  # Inshape for the dataset
    train_ds = dataset(os.path.join(data_fname, 'train'), args.nclasses, args.ntasks, embedding_idx,direction, args.nargs, nexamples=nsamples_train, split=True)
    # If normalize_image is True the mean of the dataset is subtracted from every image.
    batch_size = args.bs
    if args.normalize_image:
        train_dl = DataLoader(train_ds, batch_size=args.bs, num_workers=args.workers, shuffle=True, pin_memory=True)
        mean_image = retrieve_mean_image(train_dl, inshape, inputs_to_struct, data_fname, True)
    else:
        mean_image = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
        test_sampler = None
    else:
        train_sampler = None
        test_sampler = None
        val_sampler = None
        batch_size = args.bs * ubs
    # Creating the dataset
    train_ds = dataset(os.path.join(data_fname, 'train'), args.nclasses, args.ntasks, embedding_idx,direction, args.nargs,
                       nexamples=nsamples_train, split=True, mean_image=mean_image)
    test_ds = dataset(os.path.join(data_fname, 'test'), args.nclasses, args.ntasks, embedding_idx,direction, args.nargs,
                      nexamples=nsamples_test, split=True, mean_image=mean_image)
    if use_val:
     val_ds = dataset(os.path.join(data_fname, 'val'), args.nclasses, args.ntasks, embedding_idx, direction, args.nargs, nexamples=nsamples_test, split=True, mean_image=mean_image)
    else:
     val_ds = None
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=args.workers, shuffle=True, pin_memory=True,
                          sampler=train_sampler)
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=args.workers, shuffle=False, pin_memory=True,
                         sampler=test_sampler)
    if use_val:
     val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=args.workers, shuffle=False, pin_memory=True,  sampler=test_sampler)
    else:
     val_dl = None
    nbatches_train = len(train_dl)
    nbatches_test = len(test_dl)
    if use_val:
     nbatches_val = len(val_dl)
    else:
     nbatches_val = 0
    train_dataset = WrappedDataLoader(train_dl, preprocess)
    test_dataset = WrappedDataLoader(test_dl, preprocess)
    if use_val:
     val_dataset = WrappedDataLoader(val_dl, preprocess)
    else:
     val_dataset = None

    the_train_dataset = DatasetInfo(True, train_dataset, nbatches_train, 'Train', args.checkpoints_per_epoch,   train_sampler)
    the_test_dataset = DatasetInfo(False, test_dataset, nbatches_test, 'Test', 1, test_sampler)
    the_datasets = [the_train_dataset, the_test_dataset]
    if use_val:
     the_val_dataset = DatasetInfo(False, val_dataset, nbatches_val, 'Val', 1, val_sampler)
    if use_val:
     the_datasets.append(the_val_dataset)
    # Storing tht parameters into the Parser
    args.img_channels = 3
    args.image_size = image_size
    args.inshape = (args.img_channels, *args.image_size)
    args.ubs = ubs
    args.nbatches_train = nbatches_train
    args.nbatches_val = nbatches_val
    args.nbatches_test = nbatches_test

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

    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=args.workers, shuffle=True, pin_memory=True,
                          sampler=train_sampler)
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=args.workers, shuffle=False, pin_memory=True,
                         sampler=test_sampler)
    nbatches_train = len(train_dl)
    nbatches_test = len(test_dl)
    train_dataset = WrappedDataLoader(train_dl, preprocess)
    test_dataset = WrappedDataLoader(test_dl, preprocess)
    the_train_dataset = DatasetInfo(True, train_dataset, nbatches_train, 'Train', args.checkpoints_per_epoch,
                                    train_sampler)
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
