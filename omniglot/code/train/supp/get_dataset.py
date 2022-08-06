import os
import pickle
import torch
import argparse
from torch.utils.data import DataLoader
from supp.FlagAt import *
from supp.training_functions import *
from supp.data_functions import *
flag = True
if flag:
    from supp.omniglot_dataset import OmniglotDatasetLabelSingleTaskRight as dataset, inputs_to_struct as inputs_to_struct
else:
    from supp.omniglot_dataset import OmniglotDatasetLabelSingleTask as dataset, inputs_to_struct as inputs_to_struct


def get_dataset(args: argparse,embedding_idx: int,  data_fname: str) -> list:
    """
    returns the datasets needed for the training according to the data_fname,embedding_idx
    :param embedding_idx: the embedding for the task,creates the flag according to the embedding.
    :param args: crated the dataset according to the args
    :param data_fname: root to the files
    :return: List of the datasets,train_dl,test_dl.
    """
    path_fname = os.path.join(data_fname, 'conf')
    # Opening the conf file and retrieve number of samples, img shape,number of objects per image.
    with open(path_fname, "rb") as new_data_file:
        nsamples_train, nsamples_test, nsamples_val, nclasses, _ ,IMAGE_SIZE, num_rows_in_the_image, obj_per_row, num_chars_per_image,ndirections, valid_classes= pickle.load( new_data_file)
    # If number of gpus>1 creating larger batch size.
    ubs = args.ubs  # unified batch scale
    if args.num_gpus > 1:
        ubs = ubs * num_gpus
    inshape = (3, *IMAGE_SIZE)  # Inshape for the dataset
    train_ds = dataset(os.path.join(data_fname, 'train'), args, embedding_idx,  nexamples = nsamples_train, split=True)
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
        batch_size = args.bs * ubs
    # Creating the dataset
   # train_ds = dataset(os.path.join(data_fname, 'train'), nclasses, args.ntasks, embedding_idx, args.nargs,
   #                    nexamples=nsamples_train, split=True, mean_image=mean_image)
    train_ds = dataset(os.path.join(data_fname, 'train'), args, embedding_idx, nexamples=nsamples_train, split=True,mean_image = mean_image)

    test_ds = dataset(os.path.join(data_fname, 'test'), args, embedding_idx, nexamples=nsamples_test, split=True, mean_image = mean_image)

    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=args.workers, shuffle=True, pin_memory=True,   sampler=train_sampler)
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=args.workers, shuffle=False, pin_memory=True,
                         sampler=test_sampler)
    nbatches_train = len(train_dl)
    nbatches_test = len(test_dl)
    train_dataset = WrappedDataLoader(train_dl, preprocess)
    test_dataset = WrappedDataLoader(test_dl, preprocess)
    the_train_dataset = DatasetInfo(True, train_dataset, nbatches_train, 'Train', args.checkpoints_per_epoch,   train_sampler)
    the_test_dataset = DatasetInfo(False, test_dataset, nbatches_test, 'Test', 1, test_sampler)
    the_datasets = [the_train_dataset, the_test_dataset]
    # Storing tht parameters into the Parser
    args.img_channels = 3
    args.IMAGE_SIZE = IMAGE_SIZE
    args.inshape = (args.img_channels, *args.IMAGE_SIZE)
    args.ubs = ubs
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

