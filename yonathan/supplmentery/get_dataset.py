import os
import pickle
import torch
from torch.utils.data import DataLoader
# from supp.FlagAt import *
# from supp.training_functions import *
# from supp.data_functions import *
num_gpus=torch.cuda.device_count()

#'/home/sverkip/data/Fashion_spatial/data/samples/6_extended'

def get_dataset(direction,args,data_fname):
    conf_path=os.path.join(data_fname,'conf')
    with open(conf_path, "rb") as conf_path:
        nsamples_train, nsamples_test, nsamples_val, nclasses_existence, img_channels, LETTER_SIZE, IMAGE_SIZE, ntypes, edge_class, not_available_class, total_rows, obj_per_row, sample_nchars, ngenerate, ndirections, exclude_percentage, valid_classes, cl2let = pickle.load(
            conf_path)
        nclasses = int(max(ntypes))
    from supp.emnist_dataset import EMNISTAdjDatasetNew2 as dataset,inputs_to_struct as inputs_to_strcut
    args.inputs_to_struct=inputs_to_struct
    ndirections = 4
    train_ds = dataset(os.path.join(data_fname, 'train'), nclasses_existence, ndirections,  nexamples=nsamples_train, split=True, direction=direction)

    normalize_image = True #TODO-CHANGE
    if normalize_image:
        # just for getting the mean image
        train_dl = DataLoader(train_ds, batch_size=args.bs, num_workers=args.workers, shuffle=True, pin_memory=True)
        mean_image = retrieve_mean_image(train_dl, args.inshape, inputs_to_struct, data_fname, True)
        train_ds = dataset(os.path.join(data_fname, 'train'), nclasses_existence, ndirections, nexamples=nsamples_train, split=True, mean_image=mean_image, direction=direction)
    else:
        mean_image = None
    test_ds = dataset(os.path.join(data_fname, 'test'), nclasses_existence, ndirections, nexamples=nsamples_test,
                      split=True, mean_image=mean_image, direction=direction)
    val_ds = dataset(os.path.join(data_fname, 'val'), nclasses_existence, ndirections, nexamples=None, split=True,
                     mean_image=mean_image, direction=direction)


    nsamples_val = len(val_ds)  # validation set is only sometimes present so nsamples_val is not always available

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
        if False:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_ds, shuffle=False)
        else:
            # Although we can divide the test set across the GPUs, currently we don't gather the accuracies but only use the
            # first node in order to report the epoch's accuracy and store the optimum accuracy. Therefore only part of the test
            # set would be used for this purpose which is inaccurate (Each GPU has a correct accuracy but
            # they are not averaged into a single one).
            # Similarly for the train and validation accuracies but we can live with that drawback.
            # Therefore at least for the test set we don't split the data. Each GPU tests the whole set
            # (redundant but at least this is accurate)
            # The train loss is also inaccurate but the most important things are the gradients which are accumulated and therefore work well.
            test_sampler = None
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None
        val_sampler = None
        ubs=1
        batch_size = args.bs * ubs

    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=args.workers, shuffle=True,
                          pin_memory=True, sampler=train_sampler)
    # train_dl = DataLoader(train_ds, batch_size=batch_size,num_workers=0,shuffle=False,pin_memory=True, sampler=train_sampler)
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=args.workers, shuffle=False, pin_memory=True,
                         sampler=test_sampler)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=args.workers, shuffle=False, pin_memory=True,sampler=val_sampler)

    nbatches_train = len(train_dl)
    nbatches_val = len(val_dl)
    nbatches_test = len(test_dl)
    args.nbatches_train = nbatches_train
    args.nbatches_test = nbatches_test
    args.nbatches_val = nbatches_val
    train_dataset = WrappedDataLoader(train_dl, preprocess)
    test_dataset = WrappedDataLoader(test_dl, preprocess)
    val_dataset = WrappedDataLoader(val_dl, preprocess)

    the_train_dataset = DatasetInfo(True, train_dataset, nbatches_train, 'Train', args.checkpoints_per_epoch, train_sampler)
    the_test_dataset = DatasetInfo(False, test_dataset, nbatches_test, 'Test', 1, test_sampler)
    the_datasets = [the_train_dataset, the_test_dataset]
    if nsamples_val > 0:
        the_val_dataset = DatasetInfo(False, val_dataset, nbatches_val, 'Validation', 1, val_sampler)
        the_datasets += [the_val_dataset]
    #
    return [the_datasets, train_dl, test_dl,val_dl, train_dataset, test_dataset]


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