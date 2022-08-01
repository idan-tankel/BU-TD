# %% general initialization
import os
import torch
import torch.optim as optim
import v26.cfg as cfg
from v26.Configs.Config import Config
from v26.functions.inits import add_arguments_to_parser, init_model_options
from v26.models.DatasetInfo import DatasetInfo
from v26.models.WrappedDataLoader import WrappedDataLoader
from v26.models.BU_TD_Models import BUModelSimple, BUTDModelShared
from v26.models.Measurements import Measurements
from v26.accuracy_funcs import multi_label_accuracy, get_bounding_box, multi_label_accuracy_weighted_loss
from v26.funcs import *
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from v26.functions.loses import *


# %% visualize predictions
def from_network_transpose(samples, outs, normalize_image, mean_image, model_opts):
    if normalize_image:
        samples.image += mean_image
    samples.image = samples.image.transpose(0, 2, 3, 1)
    samples.seg = samples.seg.transpose(0, 2, 3, 1)
    if model_opts.use_td_loss:
        outs.td_head = outs.td_head.transpose(0, 2, 3, 1)
    return samples, outs


def init_train_options(model, args, num_gpus, scale_batch_size, ubs, batch_size, nbatches_train, nbatches_val,
                       nbatches_test, model_opts, train_sampler):
    train_opts = SimpleNamespace()
    train_opts.model = model
    train_opts.weight_decay = args.wd
    train_opts.initial_lr = args.lr
    learning_rates_mult = np.ones(300)
    learning_rates_mult = get_multi_gpu_learning_rate(learning_rates_mult,
                                                      num_gpus, scale_batch_size,
                                                      ubs)
    if args.checkpoints_per_epoch > 1:
        learning_rates_mult = np.repeat(learning_rates_mult,
                                        args.checkpoints_per_epoch)
    train_opts.batch_size = batch_size
    train_opts.nbatches_train = nbatches_train  # just for logging
    train_opts.nbatches_val = nbatches_val  # just for logging
    train_opts.nbatches_test = nbatches_test  # just for logging
    train_opts.num_gpus = num_gpus
    train_opts.EPOCHS = len(learning_rates_mult)
    train_opts.learning_rates_mult = learning_rates_mult
    train_opts.load_model_if_exists = True
    train_opts.model_dir = model_opts.model_dir
    train_opts.save_model = True
    train_opts.abort_after_epochs = 0
    if args.SGD:
        optimizer = optim.SGD(model.parameters(),
                              lr=train_opts.initial_lr,
                              momentum=0.9,
                              weight_decay=train_opts.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(),
                               lr=train_opts.initial_lr,
                               weight_decay=train_opts.weight_decay)
    train_opts.optimizer = optimizer
    train_opts.loss_fun = loss_fun
    lmbda = lambda epoch: train_opts.learning_rates_mult[epoch]
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)
    train_opts.scheduler = scheduler
    train_opts.checkpoints_per_epoch = args.checkpoints_per_epoch
    train_opts.train_sampler = train_sampler
    train_opts.distributed = args.distributed
    train_opts.first_node = True
    train_opts.gpu = None
    return train_opts


def create_data_loaders(train_ds, test_ds, val_ds,
                        batch_size, args, train_sampler):
    train_dl = DataLoader(train_ds,
                          batch_size=batch_size,
                          num_workers=args.workers,
                          shuffle=(train_sampler is None),
                          pin_memory=True,
                          sampler=train_sampler)
    # train_dl = DataLoader(train_ds, batch_size=batch_size,num_workers=0,shuffle=False,pin_memory=True, sampler=train_sampler)
    test_dl = DataLoader(test_ds,
                         batch_size=batch_size,
                         num_workers=args.workers,
                         shuffle=False,
                         pin_memory=True)
    val_dl = DataLoader(val_ds,
                        batch_size=batch_size,
                        num_workers=args.workers,
                        shuffle=False,
                        pin_memory=True)
    return train_dl, test_dl, val_dl


def init_datasets(inshape, flag_size, nclasses_existence, nsamples_train, nsamples_test, base_samples_dir,
                  nfeatures, batch_size, dummyds, args, flag_at):
    # from v26.models.AutoSimpleNamespace import inputs_to_struct_raw_label_all as inputs_to_struct
    from v26.avatar_dataset import AvatarDetailsDatasetLabelAll as dataset
    normalize_image = False
    train_dl = mean_image = None
    if dummyds:
        from v26.avatar_dataset import AvatarDetailsDatasetDummy as dataset
        # from v26.models.AutoSimpleNamespace import inputs_to_struct_raw as inputs_to_struct

        def flag_to_comp(flag):
            return 1, 1

        train_ds = dataset(inshape, flag_size, nclasses_existence, nsamples_train)
        test_ds = dataset(inshape, flag_size, nclasses_existence, nsamples_test)
        val_ds = dataset(inshape, flag_size, nclasses_existence, nsamples_test)
        nsamples_val = len(
            val_ds
        )  # validation set is only sometimes present so nsamples_val is not always available
    else:
        from v26.avatar_dataset import AvatarDetailsDatasetRawNew as dataset
        if flag_at is FlagAt.NOFLAG:
            from v26.models.AutoSimpleNamespace import inputs_to_struct_raw_label_all as inputs_to_struct
        else:
            from v26.models.AutoSimpleNamespace import inputs_to_struct_raw as inputs_to_struct
        train_ds = dataset(os.path.join(base_samples_dir, 'train'),
                           nclasses_existence,
                           nfeatures,
                           nsamples_train,
                           split=True)
        normalize_image = True
        if normalize_image:
            # just for getting the mean image
            train_dl = DataLoader(train_ds,
                                  batch_size=batch_size,
                                  num_workers=args.workers,
                                  shuffle=False,
                                  pin_memory=True)
            mean_image = get_mean_image(train_dl, inshape, inputs_to_struct)
            train_ds = dataset(os.path.join(base_samples_dir, 'train'),
                               nclasses_existence,
                               nfeatures,
                               nsamples_train,
                               split=True,
                               mean_image=mean_image)
        else:
            mean_image = None
        test_ds = dataset(os.path.join(base_samples_dir, 'test'),
                          nclasses_existence,
                          nfeatures,
                          nsamples_test,
                          split=True,
                          mean_image=mean_image)
        val_ds = dataset(os.path.join(base_samples_dir, 'val'),
                         nclasses_existence,
                         nfeatures,
                         split=True,
                         mean_image=mean_image)
        nsamples_val = len(
            val_ds
        )  # validation set is only sometimes present so nsamples_val is not always available

        def flag_to_comp(flag):
            avatar_id = flag[:nclasses_existence].nonzero()[0][0]
            feature_id = flag[nclasses_existence:].nonzero()[0][0]
            return avatar_id, feature_id
    return inputs_to_struct, flag_to_comp, train_ds, test_ds, val_ds, nsamples_val, normalize_image, train_dl, mean_image


def main():
    config: Config = Config()

    # running interactively uses a single GPU and plots the training results in a window
    cfg.gpu_interactive_queue = config.Visibility.interactive_session
    if config.Visibility.interactive_session:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    ngpus_per_node = torch.cuda.device_count()

    parser = argparse.ArgumentParser()

    add_arguments_to_parser(parser)
    if config.Visibility.interactive_session:
        sys.argv = ['']
    args = parser.parse_args()
    args.distributed = config.RunningSpecs.distributed
    args.hyper_search = args.hyper > -1

    if args.extended:
        base_tf_records_dir = 'extended'
    else:
        base_tf_records_dir = 'sufficient'

    init_some_hyper_params(args)

    base_samples_dir, data_fname, results_dir = init_folders(base_tf_records_dir, config)

    flag_at = config.RunningSpecs.FlagAt
    # when True use a dummy dataset instead of a real one (for debugging)
    # %% load samples
    IMAGE_SIZE, img_channels, nclasses_existence, nfeatures, nsamples_test, nsamples_train, ntypes = load_samples(
        config, data_fname)
    # %% dataset
    batch_size, flag_size, inshape, num_gpus, scale_batch_size, ubs = datasets_specs(IMAGE_SIZE, args, img_channels,
                                                                                     nclasses_existence, nfeatures,
                                                                                     ngpus_per_node)

    flag_to_comp, inputs_to_struct, normalize_image, the_datasets, train_dataset, val_dataset, mean_image, \
    nbatches_train, nbatches_val, nbatches_test, train_sampler = init_all_datasets(args, base_samples_dir, batch_size,
                                                                                   config, flag_size, inshape,
                                                                                   nclasses_existence, nfeatures,
                                                                                   nsamples_test, nsamples_train, ubs,
                                                                                   flag_at)
    # %% model options
    model_opts = init_model_opts_2(args, base_tf_records_dir, config, flag_at, flag_size, inputs_to_struct, inshape,
                                   nclasses_existence, normalize_image, ntypes, results_dir)

    log_init(model_opts)
    print_info(model_opts)
    save_script(model_opts)

    if args.hyper_search:
        logger.info('sysargv %s' % sys.argv)
        logger.info('using command options lr,bs,wd:%f,%f,%f' %
                    (args.lr, args.bs, args.wd))

    # %% create model
    model = create_model(model_opts)

    set_datasets_measurements(the_datasets, Measurements, model_opts, model)

    model_loss_and_accuracy(model_opts)

    # %% fit

    cudnn.benchmark = True

    train_opts = init_train_options(model, args, num_gpus, scale_batch_size, ubs, batch_size, nbatches_train,
                                    nbatches_val, nbatches_test, model_opts, train_sampler)

    if config.RunningSpecs.isFit:
        fit(train_opts, the_datasets)
    if not config.Visibility.interactive_session:
        sys.exit()
        # return

    load_model(train_opts,
               os.path.join(model_opts.model_dir, 'model_latest.pt'))  # TODO - fix: load latest model - missing

    fig, n, outs, preds, samples = go_over_samples_variables(inputs_to_struct, mean_image, model, model_opts,
                                                             normalize_image, train_dataset, train_opts)

    for k in range(len(samples.image)):
        go_over_sample(config.Strings.features_strings, fig, flag_to_comp, k, model_opts, n, nfeatures, outs, preds,
                       samples)
    # %% percent correct
    accs_id = [[] for i in range(nfeatures)]
    npersons = 6
    perc = np.zeros((npersons, nfeatures))
    lens = np.zeros((npersons, nfeatures))
    for inputs in val_dataset:
        loss, outs = test_step(inputs, train_opts)
        samples, outs = from_network(inputs, outs, model.module, inputs_to_struct)
        for k in range(len(samples.image)):
            flag = samples.flag[k]
            avatar_id, feature_id = flag_to_comp(flag)

            if model_opts.flag_at is FlagAt.NOFLAG:
                pred = outs.task[k].argmax(axis=0)
                pred = np.array(pred).reshape((-1, nfeatures))
                predicted_feature_value = pred[avatar_id][feature_id]
                gt = samples.label_task[k]
                gt = np.array(gt).reshape((-1, nfeatures))
                feature_value = gt[avatar_id][feature_id]
            else:
                feature_value = samples.label_task[k][0]
                predicted_feature_value = outs.task[k].argmax()
            acc_id = int(predicted_feature_value == feature_value)
            accs_id[feature_id].append(acc_id)
            perc[avatar_id][feature_id] += acc_id
            lens[avatar_id][feature_id] += 1


def go_over_samples_variables(inputs_to_struct, mean_image, model, model_opts, normalize_image, train_dataset,
                              train_opts):
    ds_iter = iter(train_dataset)
    inputs = next(ds_iter)
    loss, outs = test_step(inputs, train_opts)  # Here pass it through the network
    samples, outs = from_network(inputs, outs, model.module, inputs_to_struct)
    samples, outs = from_network_transpose(samples, outs, normalize_image, mean_image, model_opts)
    preds = np.array(outs.occurence > 0, dtype=np.float)
    fig = plt.figure(figsize=(15, 4))
    n = number_of_subplots(model_opts)
    return fig, n, outs, preds, samples


def go_over_sample(features_strings, fig, flag_to_comp, k, model_opts, n, nfeatures, outs, preds, samples):
    clear_fig(fig)
    # we only have existence information about all the avatars, without order
    present = str(samples.label_existence[k].nonzero()[0].tolist())
    tit = 'Present: %s\n' % present
    fl = samples.flag[k]
    avatar_id, feature_id = flag_to_comp(fl)
    feature_value, predicted_feature_value = predict(avatar_id, feature_id, k, nfeatures, outs, samples, model_opts)
    ins = 'Instruction: Avatar %d, %s' % (avatar_id,
                                          features_strings[feature_id])
    tit = tit + ins
    plt.imshow(samples.image[k].astype(np.uint8))
    plt.title(tit)
    ax = plt.subplot(1, n, 1)
    ax.axis('off')
    ax = plt.subplot(1, n, 2)
    ax.axis('off')
    plt.imshow(samples.image[k].astype(np.uint8))
    curseg = samples.seg[k].squeeze()
    # somewhat hacky way to find the background: assuming it is the most dominant feature
    curseg_ch0 = curseg[:, :, 0]
    vals, counts = np.unique(curseg_ch0, return_counts=True)
    curseg_min = vals[counts.argmax()]
    mask = curseg > curseg_min
    add_mask(ax, mask)
    gt_str, pred_str = pred_and_gt_text(feature_id, feature_value, features_strings, predicted_feature_value)
    predicted_existing_avatar_ids = preds[k]
    font = get_font(feature_value, predicted_feature_value)
    title_with_td_loss(font, gt_str, k, model_opts, n, outs, pred_str)
    print(k)
    print(predicted_existing_avatar_ids)
    #    savefig(os.path.join(avatars_dir, 'examples%d.png'% k), dpi=90, bbox_inches='tight' )
    pause_image()


def title_with_td_loss(font, gt_str, k, model_opts, n, outs, pred_str):
    if model_opts.use_td_loss:
        tit_str = gt_str
        plt.title(tit_str)
    else:
        tit_str = gt_str + '\n' + pred_str
        plt.title(tit_str, fontdict=font)
    if model_opts.use_td_loss:
        ax = plt.subplot(1, n, 3)
        ax.axis('off')
        image_tdk = np.array(outs.td_head[k])
        image_tdk = image_tdk - np.min(image_tdk)
        image_tdk = image_tdk / np.max(image_tdk)
        plt.imshow(image_tdk)
        plt.title(pred_str, fontdict=font)


def get_font(feature_value, predicted_feature_value):
    if feature_value == predicted_feature_value:
        font = {'color': 'blue'}
    else:
        font = {'color': 'red'}
    return font


def pred_and_gt_text(feature_id, feature_value, features_strings, predicted_feature_value):
    gt_str = 'Ground Truth: %s = %d' % (features_strings[feature_id],
                                        feature_value)
    pred_str = 'Prediction: %s = %d' % (features_strings[feature_id],
                                        predicted_feature_value)
    return gt_str, pred_str


def add_mask(ax, mask):
    if len(np.unique(mask)) > 1:
        [stx, sty, endx, endy] = get_bounding_box(mask)
        ax.add_patch(
            patches.Rectangle((stx, sty),
                              endx - stx,
                              endy - sty,
                              linewidth=2,
                              edgecolor='g',
                              facecolor='none'))


def predict(avatar_id, feature_id, k, nfeatures, outs, samples, model_opts):
    if model_opts.flag_at is FlagAt.NOFLAG:
        pred = outs.task[k].argmax(axis=0)
        pred = np.array(pred).reshape((-1, nfeatures))
        predicted_feature_value = pred[avatar_id][feature_id]
        gt = samples.label_task[k]
        gt = np.array(gt).reshape((-1, nfeatures))
        feature_value = gt[avatar_id][feature_id]
    else:
        feature_value = samples.label_task[k][0]
        predicted_feature_value = outs.task[k].argmax()
    return feature_value, predicted_feature_value


def clear_fig(fig):
    fig.clf()
    fig.tight_layout()


def number_of_subplots(model_opts) -> int:
    if model_opts.use_td_loss:
        n = 3
    else:
        n = 2
    return n


def model_loss_and_accuracy(model_opts):
    if model_opts.flag_at is FlagAt.NOFLAG:
        # if not model_opts.head_of_all_features:  # TODO - change it from config
        model_opts.bu2_loss = multi_label_loss_weighted_loss
        # else:
        #     model_opts.bu2_loss = multi_label_loss_weighted_loss_only_one
        model_opts.task_accuracy = multi_label_accuracy_weighted_loss
    else:
        model_opts.bu2_loss = multi_label_loss
        model_opts.task_accuracy = multi_label_accuracy


def datasets_specs(IMAGE_SIZE, args, img_channels, nclasses_existence, nfeatures, ngpus_per_node):
    inshape = (img_channels, *IMAGE_SIZE)
    flag_size = nclasses_existence + nfeatures
    num_gpus = ngpus_per_node
    batch_size = args.bs
    scale_batch_size = 1
    ubs = scale_batch_size  # unified batch scale
    if num_gpus > 1:
        ubs = ubs * num_gpus
    return batch_size, flag_size, inshape, num_gpus, scale_batch_size, ubs


def init_folders(base_tf_records_dir, config):
    avatars_dir = os.path.join(config.Folders.data_dir, config.Folders.avatars)
    new_avatars_dir = os.path.join(avatars_dir, config.Folders.samples)
    base_samples_dir = os.path.join(new_avatars_dir, base_tf_records_dir)
    data_fname = os.path.join(base_samples_dir, config.Folders.conf)
    results_dir = os.path.join(config.Folders.data_dir, config.Folders.results)
    return base_samples_dir, data_fname, results_dir


def create_model(model_opts):
    if model_opts.flag_at is FlagAt.BU1_SIMPLE:
        model = BUModelSimple(model_opts)
    else:

            model = BUTDModelShared(model_opts)
    if not torch.cuda.is_available():
        logger.info('using CPU, this will be slow')
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
    set_model(model)
    set_model_opts(model_opts)
    return model


def init_some_hyper_params(args):
    if args.hyper_search:
        index = args.hyper
        import itertools

        if args.SGD:
            lrs = [0.001, 0.0001]
            bss = [10]
            wds = [0.0001, 0.0002]
            cmd_options = list(itertools.product(lrs, bss, wds))
            cmd_options = np.array(cmd_options)
        else:
            lrs = np.array([0.0005, 0.001, 0.002])
            bss = [10]
            wds = [0.0001, 0.0002]
            cmd_options = list(itertools.product(lrs, bss, wds))
            cmd_options = np.array(cmd_options)
        args.lr, args.bs, args.wd = cmd_options[index]
        args.bs = int(args.bs)


def init_model_opts_2(args, base_tf_records_dir, config: Config, flag_at, flag_size,
                      inputs_to_struct,
                      inshape, nclasses_existence, normalize_image, ntypes, results_dir):
    model_opts = init_model_options(config, flag_at, normalize_image, nclasses_existence, ntypes, flag_size,
                                    BatchNorm, inshape)
    flag_str = str(model_opts.flag_at).replace('.', '_').lower()
    num_gpus = torch.cuda.device_count()
    dummytype = '_dummyds' * config.Datasets.dummyds
    base_model_dir = 'avatar_details_pyt_v26_%s_sgd%d%s' % (flag_str, args.SGD,
                                                            dummytype)
    if not model_opts.use_td_loss and model_opts.use_bu1_loss:
        base_model_dir = base_model_dir + '_two_losses'
    if model_opts.head_of_all_features:
        base_model_dir = base_model_dir + '_all_features'
    model_dir = os.path.join(results_dir,
                             base_model_dir + '_%s' % (base_tf_records_dir))
    if args.hyper_search:
        model_dir = os.path.join(model_dir, 'cmdo%d' % args.hyper)
        if args.only_cont and not os.path.exists(model_dir):
            # return
            sys.exit()
    model_opts.inputs_to_struct = inputs_to_struct
    # just for logging. changes must be made in code in order to take effect
    model_opts.lateral_per_neuron = False
    model_opts.separable = True
    model_opts.model_dir = model_dir
    if config.Logging.enable_logging:
        model_opts.logfname = 'log.txt'
    else:
        print('Logging disabled')
        model_opts.logfname = None
    os.makedirs(model_dir, exist_ok=True)
    model_opts.distributed = args.distributed

    return model_opts


def init_all_datasets(args, base_samples_dir, batch_size, config, flag_size, inshape, nclasses_existence, nfeatures,
                      nsamples_test, nsamples_train, ubs, flag_at):
    inputs_to_struct, flag_to_comp, train_ds, test_ds, val_ds, nsamples_val, normalize_image, train_dl, mean_image = init_datasets(
        inshape, flag_size, nclasses_existence, nsamples_train, nsamples_test, base_samples_dir,
        nfeatures, batch_size, config.Datasets.dummyds, args, flag_at)
    set_inputs_to_struct(inputs_to_struct=inputs_to_struct)
    train_sampler = None
    batch_size = batch_size * ubs
    train_dl, test_dl, val_dl = create_data_loaders(train_ds, test_ds, val_ds, batch_size, args, train_sampler)
    nbatches_train = len(train_dl)
    nbatches_val = len(val_dl)
    nbatches_test = len(test_dl)
    train_dataset = WrappedDataLoader(train_dl, preprocess)
    test_dataset = WrappedDataLoader(test_dl, preprocess)
    val_dataset = WrappedDataLoader(val_dl, preprocess)
    the_train_dataset = DatasetInfo(True, train_dataset, nbatches_train, 'Train',
                                    args.checkpoints_per_epoch)
    the_test_dataset = DatasetInfo(False, test_dataset, nbatches_test, 'Test')
    the_datasets = [the_train_dataset, the_test_dataset]
    if nsamples_val > 0:
        the_val_dataset = DatasetInfo(False, val_dataset, nbatches_val,
                                      'Validation')
        the_datasets += [the_val_dataset]
    return flag_to_comp, inputs_to_struct, normalize_image, the_datasets, train_dataset, val_dataset, mean_image, nbatches_train, nbatches_val, nbatches_test, train_sampler


def load_samples(config, data_fname):
    if not config.Datasets.dummyds:
        with open(data_fname, "rb") as new_data_file:
            nsamples_train, nsamples_test, nsamples_val, nfeatures, nclasses, nclasses_existence, ntypes, img_channels, IMAGE_SIZE = pickle.load(
                new_data_file)
    else:
        nsamples_train = 200
        nsamples_test = 200
        nclasses_existence = 6
        # if config.Models.features == 'all':
        #     nclasses_existence = 42
        nfeatures = 7
        ntypes = [9]
        IMAGE_SIZE = [224, 448]
        img_channels = 3
    return IMAGE_SIZE, img_channels, nclasses_existence, nfeatures, nsamples_test, nsamples_train, ntypes


if __name__ == "__main__":
    main()
