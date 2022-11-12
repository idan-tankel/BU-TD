# # %% ipython init
# # %load_ext autoreload
# # %autoreload 2
# a = get_ipython()
# #a.magic('%load_ext autoreload')
# #a.magic('%autoreload 2')
# a.magic("%config ZMQInteractiveShell.ast_node_interactivity='last_expr_or_assign'")
# a.magic('cd ~/code/counter_stream')
#TODO:
    # improve checkpoints-per-epoch
    # in ddp gradients are averaged, not summed?
    # is to() same as cuda(args.gpu). each process have in theory acess to all visible gpus
    # large storage
# %% general initialization
import os
import v27.cfg as cfg
# three ways of running: running interactively uses a single GPU, running without the environment variable as a script uses DataParallel
# and running from within ..._dist.py uses DistributedDataParallel
interactive_session = 'LIAV_RUN_INTERACTIVE' in os.environ and os.environ['LIAV_RUN_INTERACTIVE']=='1'
# interactive_session = False
cfg.gpu_interactive_queue = interactive_session
is_main = __name__ == '__main__'
# is_main = False
optimize_hyperparams = False
if interactive_session:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from v27.funcs import *

def update(variables):
    globals().update(variables)

def gen_init(mytype='internal',ngpus_per_node=None, args=None,log_msgs=None):
    ENABLE_LOGGING = True
    home_dir = os.path.expanduser('~')
    cs_dir = os.path.join(home_dir,'code/counter_stream')
    data_dir = os.path.join(cs_dir,'data')
    new_ds_conf=False
    emnist_dir = os.path.join(data_dir, 'emnist')
    if new_ds_conf:
        base_tf_records_dir = ''
    else:
        base_tf_records_dir = 'food101'
    new_emnist_dir=os.path.join(emnist_dir,'samples')
    base_samples_dir=os.path.join(new_emnist_dir,base_tf_records_dir)
    data_fname=os.path.join(base_samples_dir,'conf')
    results_dir = os.path.join(emnist_dir, 'results')
    flag_at=FlagAt.TD
    # when True use a dummy dataset instead of a real one (for debugging)
    dummyds =False
    cycle_lr = False

    update(locals())

def get_args(config=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--hyper', default=-1, type=int)
    parser.add_argument('--only_cont', action='store_true')
    parser.add_argument('--SGD', action='store_true')
    parser.add_argument('-lr', default=1e-4, type=float)
    parser.add_argument('-bs', default=64, type=int)
    parser.add_argument('-wd', default=0.0001, type=float)
    parser.add_argument('-opt', default='ADAM', type=str)
    parser.add_argument('--avg-grad', action='store_true')
    parser.add_argument('--checkpoints-per-epoch', default=1,type=int)
    # if interactive_session and not is_main:
    sys.argv=['']
    args = parser.parse_args()
    args.gpu = None
    args.distributed = False
    args.multiprocessing_distributed = False
    args.hyper_search = args.hyper > -1
    if config:
        args.lr = config['lr']
        args.wd = config['wd']
        args.SGD = False
        args.bs = config['bs']
        args.opt = config['opt']
    args.optimize_hyperparams = optimize_hyperparams
    return args

if interactive_session or is_main:
    gen_init()
    ngpus_per_node = torch.cuda.device_count()
    args = get_args()
    log_msgs=[]

# %% load samples
def load_samples():
    nclasses = 102
    ntypes = nclasses * np.ones(1,dtype=int)
    IMAGE_SIZE = [224,224]
    img_channels = 3

    update(locals())

if interactive_session or is_main:
    load_samples()
# %% dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
# from torchvision.datasets import Flowers102
from torchvision.datasets import Food101 
class ClassificationDataset(Food101):

    def __init__(self, image_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_size = image_size

    def __getitem__(self, index):
        image,label = super().__getitem__(index)
        image = transforms.ToTensor()(image)
        image = transforms.Resize(self.image_size)(image)
        image = 255*image
        label_task = torch.tensor(label)
        label_task = label_task.view((-1))
        return image,label_task


def dataset():
    inshape = (img_channels,*IMAGE_SIZE)

    num_gpus = ngpus_per_node
    batch_size = args.bs
    scale_batch_size = 1
    ubs = scale_batch_size # unified batch scale
    if num_gpus > 1:
        ubs = ubs * num_gpus

    
    def inputs_to_struct(inputs):
        image,label_task = inputs
        sample = AutoSimpleNamespace(locals(), image,label_task).tons()
        return sample
   
    dataset = ClassificationDataset
    train_ds = dataset(IMAGE_SIZE,base_samples_dir,'train',download=True)
    normalize_image =False 
    if normalize_image:
        # just for getting the mean image
        train_dl = DataLoader(train_ds, batch_size=batch_size,num_workers=args.workers,shuffle=False,pin_memory=True)
        # mean_image = retrieve_mean_image(train_dl,inshape,inputs_to_struct, base_samples_dir, True)
        mean_image= get_mean_image(train_dl, inshape,inputs_to_struct, stop_after=1000)
        train_ds = CachedDataset(dataset(os.path.join(base_samples_dir,'train'),nclasses_existence, ndirections, nexamples = nsamples_train,split = True,mean_image = mean_image), cache_supplier=Cache(shuffle=True,shuffle_type = args.distributed,num_gpus=num_gpus,cache_postprocess_fun=cache_postprocess_img) if use_cache else None)
    else:
        mean_image = None
    test_ds = dataset(IMAGE_SIZE,base_samples_dir,'test',download=False)
    # val_ds = dataset(base_samples_dir,'val',download=False)
    nsamples_train = len(train_ds)
    nsamples_test = len(test_ds)
    # nsamples_val = len(val_ds)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds,shuffle=True)
        if True:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_ds,shuffle=False)
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
        val_sampler = None
    else:
        train_sampler = None
        test_sampler = None
        val_sampler = None
        batch_size = batch_size * ubs

    train_dl = DataLoader(train_ds, batch_size=batch_size,num_workers=args.workers,shuffle=(train_sampler is None),pin_memory=True, sampler=train_sampler)
    # train_dl = DataLoader(train_ds, batch_size=batch_size,num_workers=0,shuffle=False,pin_memory=True, sampler=train_sampler)
    test_dl = DataLoader(test_ds, batch_size=batch_size,num_workers=args.workers,shuffle=False,pin_memory=True, sampler=test_sampler)
    # val_dl = DataLoader(val_ds, batch_size=batch_size,num_workers=args.workers,shuffle=False,pin_memory=True, sampler=val_sampler)

    nbatches_train = len(train_dl)
    # nbatches_val = len(val_dl)
    nbatches_test = len(test_dl)
    # commented out as in distributed, each train loader using the sampler will have the correct size: nsamples/(batch_size*ngpus)
    # nbatches_train = int(np.ceil(nsamples_train / batch_size))
    # nbatches_val = int(np.ceil(nsamples_test / batch_size))
    # nbatches_test = int(np.ceil(nsamples_test / batch_size))
    # print(batch_size,nbatches_train)

    train_dataset = WrappedDataLoader(train_dl, preprocess)
    test_dataset = WrappedDataLoader(test_dl, preprocess)
    # val_dataset = WrappedDataLoader(val_dl, preprocess)

    the_train_dataset = DatasetInfo(True,train_dataset,nbatches_train,'Train',args.checkpoints_per_epoch,train_sampler)
    the_test_dataset = DatasetInfo(False,test_dataset,nbatches_test,'Test',1,test_sampler)
    the_datasets = [the_train_dataset,the_test_dataset]
    # if nsamples_val>0:
    #     the_val_dataset = DatasetInfo(False,val_dataset,nbatches_val,'Validation',1,val_sampler)
    #     the_datasets += [the_val_dataset]
    # for inputs in train_dl:
    #     samples = inputs_to_struct(inputs);
    #     print(samples.id)
    # return
    if False:
        inputs = train_ds[0];
        samples = inputs_to_struct(inputs);
    # benchmark dataset
    #if False:
    #    import time
    #    start_time = time.time()
    #    for batchi,(inputs) in enumerate(train_dl):
    #        a=0
    #    duration = time.time() - start_time
    #    print(duration)
    #    logger.info(duration)
    update(locals())

if interactive_session or is_main:
    dataset()
# %% model options
def set_model_opts():
    model_opts = SimpleNamespace()
    model_opts.data_dir = data_dir
    model_opts.nclasses = [nclasses]

    model_opts.use_td_loss = False
    model_opts.use_bu1_loss = False
    model_opts.use_bu2_loss = True
    model_opts.ntaskhead_fc = 1
    model_opts.cycle_lr = cycle_lr
    model_opts.nfilters=[128]
    model_opts.inshape=inshape

    num_gpus = torch.cuda.device_count()
    dummytype = '_dummyds' * dummyds
    cyclelr_st = '_cyclr' * cycle_lr
    opt_params_st ='ray_' * optimize_hyperparams
    base_model_dir = '%sv27_vit_p16_normimg_nowd_cct4_224sz_128dim' % (opt_params_st)
    model_dir = os.path.join(results_dir,
                             base_model_dir + '_%s' % (base_tf_records_dir))
    if args.hyper_search:
        model_dir = os.path.join(model_dir,'cmdo%d'%args.hyper)
        if args.only_cont and not os.path.exists(model_dir):
            # return
            sys.exit()

    model_opts.inputs_to_struct = inputs_to_struct
    # just for logging. changes must be made in code in order to take effect
    model_opts.lateral_per_neuron = False
    model_opts.separable = True

    model_opts.model_dir = model_dir
    if ENABLE_LOGGING:
        model_opts.logfname = 'log.txt'
    else:
        print('Logging disabled')
        model_opts.logfname = None

    first_node = not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)
    if first_node:
        os.makedirs(model_dir,exist_ok=True)

    model_opts.distributed = args.distributed
    if args.distributed:
        # synchronize point so all distributed processes would have model_dir folder available
        import torch.distributed as dist
        dist.barrier()
        model_opts.module = args.module

    log_init(model_opts)
    if first_node:
        print_info(model_opts)
        save_script(model_opts)

        if args.hyper_search:
            logger.info('sysargv %s'% sys.argv)
            logger.info('using command options lr,bs,wd:%f,%f,%f' % (args.lr,args.bs,args.wd))

    for msg in log_msgs:
        logger.info(msg)
    update(locals())

if interactive_session or is_main:
    set_model_opts()
# %% create model
def create_model():
    from vit_pytorch.cct import CCT 
    
    dim= model_opts.nfilters[-1]
    class BUModelClassification(nn.Module):
    
        def __init__(self, opts):
            super(BUModelClassification, self).__init__()
            v = CCT(
                img_size = inshape[1:],
                embedding_dim = opts.nfilters[-1],
                num_classes=opts.nfilters[-1],
                n_conv_layers = 2,
                kernel_size = 3,
                stride = 2,
                padding = 3,
                pooling_kernel_size = 3,
                pooling_stride = 2,
                pooling_padding = 1,
                num_layers = 4,
                num_heads = 2,
                mlp_ratio = 1.,
            )           

            self.bumodel=v
            self.taskhead = MultiLabelHead(opts)

        def forward(self, inputs):
            samples = inputs_to_struct(inputs)
            model_inputs = samples.image

            bu_out = self.bumodel(model_inputs)
            task_out = self.taskhead(bu_out)
            return task_out
    
        def outs_to_struct(self,outs):
            task_out = outs
            outs_ns = SimpleNamespace(task=task_out)
            return outs_ns
    
    model = BUModelClassification(model_opts)

    if not torch.cuda.is_available():
        logger.info('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have

            # commented out as I prefer to use the original batch size in each GPU rather than dividing it
            # batch_size = int(batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=False)
            # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    update(locals())

if interactive_session or is_main:
    create_model()
# %% loss and metrics
def get_model_outs(model,outs):
    if type(model) is torch.nn.DataParallel or type(model) is torch.nn.parallel.DistributedDataParallel:
        return model.module.outs_to_struct(outs)
    else:
        return model.outs_to_struct(outs)

def multi_label_accuracy_base(outs,samples,nclasses):
    cur_batch_size = samples.image.shape[0]
    preds = torch.zeros((cur_batch_size,len(nclasses)),dtype=torch.int).to(dev, non_blocking=True)
    for k in range(len(nclasses)):
        taskk_out = outs.task[:,:,k]
        predsk = torch.argmax(taskk_out, axis=1)
        preds[:,k]=predsk
    label_task = samples.label_task
    task_accuracy = (preds == label_task).float()
    return preds,task_accuracy

def multi_label_accuracy(outs,samples,nclasses):
    preds,task_accuracy = multi_label_accuracy_base(outs,samples,nclasses)
    task_accuracy = task_accuracy.mean(axis=1) # per single example
    return preds,task_accuracy

def multi_label_accuracy_weighted_loss(outs,samples,nclasses):
    preds,task_accuracy = multi_label_accuracy_base(outs,samples,nclasses)
    loss_weight = samples.loss_weight
    task_accuracy = task_accuracy * loss_weight
    task_accuracy = task_accuracy.sum(axis=1)/loss_weight.sum(axis=1) # per single example
    return preds,task_accuracy

class Measurements(MeasurementsBase):
    def __init__(self, opts, model):
        super(Measurements, self).__init__(opts)
        # self.reset()
        self.model = model
        self.opts = opts
        if self.opts.use_bu1_loss:
            super().add_name('Occurence Acc')

        if self.opts.use_bu2_loss:
            super().add_name('Task Acc')

        self.init_results()

    def update(self, inputs, outs, loss):
        super().update(inputs, outs, loss)
        outs = get_model_outs(model,outs)
        samples = inputs_to_struct(inputs)
        if self.opts.use_bu1_loss:
            occurence_pred=outs.occurence>0
            occurence_accuracy = (occurence_pred == samples.label_existence).type(torch.float).mean(axis=1)
            super().update_metric(self.occurence_accuracy,occurence_accuracy.sum().cpu().numpy())

        if self.opts.use_bu2_loss:
            preds,task_accuracy = self.opts.task_accuracy(outs,samples,self.opts.nclasses)
            super().update_metric(self.task_accuracy,task_accuracy.sum().cpu().numpy())

    def reset(self):
        super().reset()
        if self.opts.use_bu1_loss:
            self.occurence_accuracy=np.array(0.0)
            self.metrics += [self.occurence_accuracy]

        if self.opts.use_bu2_loss:
            self.task_accuracy=np.array(0.0)
            self.metrics += [self.task_accuracy]

def multi_label_loss_base(outs,samples,nclasses):
    losses_task = torch.zeros((samples.label_task.shape)).to(dev, non_blocking=True)
    for k in range(len(nclasses)):
        taskk_out = outs.task[:,:,k]
        label_taskk = samples.label_task[:,k]
        loss_taskk = loss_task_multi_label(taskk_out,label_taskk)
        losses_task[:,k]=loss_taskk
    return losses_task

def multi_label_loss(outs,samples,nclasses):
    losses_task = multi_label_loss_base(outs,samples,nclasses)
    loss_task = losses_task.mean() # a single valued result for the whole batch
    return loss_task

def multi_label_loss_weighted_loss(outs, samples, nclasses):
    losses_task = multi_label_loss_base(outs, samples, nclasses)
    loss_weight = samples.loss_weight
    losses_task = losses_task * loss_weight
    loss_task = losses_task.sum()/loss_weight.sum() # a single valued result for the whole batch
    return loss_task

def loss_fun(inputs, outs):
    # nn.CrossEntropyLoss on GPU is not deterministic. However using CPU doesn't seem to help either...
    outs = get_model_outs(model,outs)
    samples = inputs_to_struct(inputs)
    losses=[]
    if model_opts.use_bu1_loss:
        loss_occ = loss_occurence(outs.occurence,samples.label_existence)
        losses.append(loss_occ)

    if model_opts.use_td_loss:
        loss_seg_td = loss_seg(outs.td_head, samples.seg)
        loss_bu1_after_convergence = 1
        loss_td_after_convergence = 100
        ratio = loss_bu1_after_convergence / loss_td_after_convergence
        losses.append(ratio * loss_seg_td)

    if model_opts.use_bu2_loss:
        loss_task = model_opts.bu2_loss(outs,samples,model_opts.nclasses)
        losses.append(loss_task)

    loss = torch.sum(torch.stack(losses))
#    print(loss_occ.item(),loss_seg_td.item(),loss_task.item())
    return loss

def loss_and_metrics():
    loss_occurence = torch.nn.BCEWithLogitsLoss(reduction='mean').to(dev)
    loss_seg = torch.nn.MSELoss(reduction='mean').to(dev)
    loss_task_op = nn.CrossEntropyLoss(reduction='mean').to(dev)
    loss_task_multi_label = nn.CrossEntropyLoss(reduction='none').to(dev)

    set_datasets_measurements(the_datasets,Measurements,model_opts,model)
    model_opts.bu2_loss = multi_label_loss
    model_opts.task_accuracy = multi_label_accuracy

    update(locals())

if interactive_session or is_main:
    loss_and_metrics()

# %% fit
def train_and_fit(checkpoint_dir=None):
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True

    train_opts = SimpleNamespace()
    train_opts.model = model
    train_opts.weight_decay = args.wd
    train_opts.initial_lr = args.lr
    learning_rates_mult = np.ones(200)
    if args.avg_grad:
        learning_rates_mult = ubs *learning_rates_mult
    learning_rates_mult = get_multi_gpu_learning_rate(learning_rates_mult,num_gpus,scale_batch_size,ubs)
    if args.checkpoints_per_epoch>1:
        learning_rates_mult = np.repeat(learning_rates_mult,args.checkpoints_per_epoch)
    train_opts.batch_size = batch_size
    train_opts.nbatches_train = nbatches_train # just for logging
    # train_opts.nbatches_val = nbatches_val # just for logging
    train_opts.nbatches_test = nbatches_test # just for logging

    train_opts.num_gpus = num_gpus
    train_opts.EPOCHS = len(learning_rates_mult)
    train_opts.learning_rates_mult = learning_rates_mult
    train_opts.model_dir=model_opts.model_dir
    train_opts.abort_after_epochs = 0
    learned_params = model.parameters()
    if args.opt=='SGD':
        optimizer = optim.SGD(learned_params, lr = train_opts.initial_lr, momentum=0.9, weight_decay=train_opts.weight_decay)
    elif args.opt=='ADAM':
        optimizer = optim.Adam(learned_params, lr = train_opts.initial_lr, weight_decay=0)
    else:
        import torch_optimizer as toptim

        if args.opt=='DiffGrad':
            optimizer = toptim.DiffGrad(learned_params,lr = train_opts.initial_lr, weight_decay=train_opts.weight_decay)
        elif args.opt=='AccSGD':
            optimizer = toptim.AccSGD(learned_params,lr = train_opts.initial_lr, weight_decay=train_opts.weight_decay)
        elif args.opt=='AdaBelief':
            optimizer = toptim.AdaBelief(learned_params,lr = train_opts.initial_lr, weight_decay=train_opts.weight_decay)
        elif args.opt=='Lamb':
            optimizer = toptim.Lamb(learned_params,lr = train_opts.initial_lr, weight_decay=train_opts.weight_decay)
        elif args.opt=='Lookahead':
            optimizer = optim.Adam(learned_params, lr = train_opts.initial_lr, weight_decay=train_opts.weight_decay)
            optimizer = toptim.Lookahead(optimizer)
        elif args.opt=='MADGRAD':
            optimizer = toptim.MADGRAD(learned_params,lr = train_opts.initial_lr, weight_decay=train_opts.weight_decay)
        elif args.opt=='QHAdam':
            optimizer = toptim.QHAdam(learned_params,lr = train_opts.initial_lr, weight_decay=train_opts.weight_decay)
        elif args.opt=='SWATS':
            optimizer = toptim.SWATS(learned_params,lr = train_opts.initial_lr, weight_decay=train_opts.weight_decay)
        elif args.opt=='RangerVA':
            optimizer = toptim.RangerVA(learned_params,lr = train_opts.initial_lr, weight_decay=train_opts.weight_decay)

    if cycle_lr:
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr = 0.0001, max_lr = 0.002, step_size_up=nbatches_train//2, step_size_down=None, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=False, last_epoch=-1)
    else:
        lmbda = lambda epoch: train_opts.learning_rates_mult[epoch]
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)
    train_opts.optimizer=optimizer
    train_opts.loss_fun = loss_fun
    train_opts.scheduler = scheduler
    train_opts.checkpoints_per_epoch=args.checkpoints_per_epoch
    train_opts.distributed = args.distributed
    train_opts.first_node = first_node
    train_opts.gpu = args.gpu
    train_opts.checkpoint_dir=checkpoint_dir
    train_opts.optimize_hyperparams = optimize_hyperparams
    if train_opts.optimize_hyperparams:
        train_opts.load_model_if_exists=False
        train_opts.save_model = False
    else:
        train_opts.load_model_if_exists= True
        train_opts.save_model = True
    
    update(locals())
    fit(train_opts,the_datasets)

if interactive_session or is_main:
    train_and_fit()

#load_model(train_opts,os.path.join(model_dir,'model_latest.pt'));
# %%
def the_main(config=None,checkpoint_dir=None):
    log_msgs=[] # in order to also log them in the file logger (which is created later)
    args = get_args(config)
    ngpus_per_node=1
    gen_init('',ngpus_per_node,args,log_msgs)
    load_samples()
    dataset()
    set_model_opts()
    create_model()
    loss_and_metrics()
    train_and_fit(checkpoint_dir)

# %% optimize_hyperparams
if optimize_hyperparams:
    import ray
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler
# sync_config=tune.SyncConfig(
#         syncer=None  # Disable syncing
#     )
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp', default='1', type=str)
    args = parser.parse_args()
    minloss = False
    use_nevergrad = False
    if not use_nevergrad:
        config = {"lr": tune.loguniform(0.0001,0.2),
              'wd': tune.choice([0.0001,0.0002]),
              'bs': tune.choice([32]),
              'opt': tune.choice(['SGD','ADAM','DiffGrad','AdaBelief','AccSGD','Lamb','Lookahead','MADGRAD','QHAdam','SWATS','RangerVA']),
            }
    else:
        # TODO also remove config = config, from tune.run. however restore doesn't work
        import nevergrad as ng
        space = ng.p.Dict(
            lr=ng.p.Scalar(lower=0.0005, upper=0.002),
            wd=ng.p.Choice(choices=[0.0001,0.0002]),
            bs=ng.p.Choice(choices=[10]),
            SGD=ng.p.Choice(choices=[False,True])
        )
    from ray.tune import ProgressReporter
    from typing import Dict, List
    from ray.tune.trial import DEBUG_PRINT_INTERVAL, Trial
    if minloss:
        metric = 'loss'
        mode = 'min'
    else:
        metric = 'accuracy'
        mode = 'max'
    scheduler = ASHAScheduler(metric=metric,mode=mode,max_t=10,grace_period=5,reduction_factor=2)

    if use_nevergrad:
        import nevergrad as ng
        from ray.tune.suggest.nevergrad import NevergradSearch
        algo = NevergradSearch(optimizer=ng.optimizers.OnePlusOne,metric=metric,mode=mode,space=space)
        from ray.tune.suggest import ConcurrencyLimiter
        algo = ConcurrencyLimiter(algo, max_concurrent=4)
    elif True:
        from ray.tune.suggest.hyperopt import HyperOptSearch
        algo = HyperOptSearch(metric=metric,mode=mode)
    else:
        from ray.tune.suggest.basic_variant import BasicVariantGenerator
        algo = BasicVariantGenerator()

    exp_name = args.exp
    ray_dir = '/home/projects/shimon/liav/ray_results'
    log_path = os.path.join(ray_dir,exp_name)
    if os.path.exists(log_path):
        resume=True
        try:
            algo.restore_from_dir(log_path)
        except:
            resume = False
    else:
        resume = False
        os.makedirs(log_path)

    log_fname =os.path.join(log_path,'log.txt')
    logger = setup_logger(log_fname)
    all_msgs=[]
    class CLIReporter2(CLIReporter):
        def report(self, trials: List[Trial], done: bool, *sys_info: Dict):
            print(self._progress_str(trials, done, *sys_info))
            all_msgs.append(self._progress_str(trials, done, *sys_info))
            logger.info(all_msgs[-1])

    reporter = CLIReporter2(
            metric_columns=["loss", "accuracy", "training_iteration"])
    tot_gpus = torch.cuda.device_count()
    ray.init(num_cpus=tot_gpus,num_gpus=tot_gpus,ignore_reinit_error=True,local_mode=False)

    result = tune.run(the_main,
            name = exp_name,
            config = config,
            resources_per_trial={"cpu": 1, "gpu": 1},
            num_samples=20,
            scheduler=scheduler,
            progress_reporter =reporter,
            search_alg=algo,
            log_to_file=True,
            resume = resume
                )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))
    best_trial = result.get_best_trial("accuracy", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))


# %% visualize predictions
if interactive_session:
    from mimshow import *
    def from_network_transpose(samples,outs):
        if normalize_image:
            samples.image+=mean_image
        samples.image=samples.image.transpose(0,2,3,1)
        return samples,outs


    ds_iter = iter(train_dataset)
    inputs=next(ds_iter)
    loss, outs = test_step(inputs,train_opts)
    samples,outs = from_network(inputs,outs,model.module,inputs_to_struct)
    samples,outs = from_network_transpose(samples,outs)
    fig = plt.figure(figsize=(10, 5))
    for k in range(len(samples.image)):
        fig.clf()
        fig.tight_layout()
        ax=plt.subplot(1, 1, 1)
        ax.axis('off')
        mimshow(samples.image[k])
        gt_val = samples.label_task[k][0]
        pred_val=outs.task[k].argmax()
        if gt_val == pred_val:
            font = {'color':  'blue'}
        else:
            font = {'color':  'red'}
        gt_str = 'Ground Truth:%s' % gt_val
        pred_str = 'Prediction:%s' % pred_val
        
        tit_str = gt_str+'\n'+pred_str
        plt.title(tit_str,fontdict=font)
        print(k)
    #    savefig(os.path.join(avatars_dir, 'examples%d.png'% k) )
        pause_image()
    # %% percent correct - task
    conf = np.zeros((nclasses, nclasses))
    for inputs in test_dataset:
        loss, outs = test_step(inputs,train_opts)
        samples,outs = from_network(inputs,outs,model.module,inputs_to_struct)
        for k in range(len(samples.image)):
            gt_val = samples.label_task[k][0]
            pred_val=outs.task[k].argmax()
            conf[gt_val][pred_val]+=1

    print(np.diag(conf).sum()/conf.sum())
    # for just the accuracy:
    # the_val_dataset.do_epoch(0,train_opts)
    # print(the_val_dataset.measurements.print_epoch())
    # %% percent correct - occurence
    accs = np.zeros(nclasses_existence)
    lens = np.zeros(nclasses_existence)
    for inputs in test_dataset:
        loss, outs = test_step(inputs,train_opts)
        samples,outs = from_network(inputs,outs,model.module,inputs_to_struct)
        for gt_value,occ_out in zip(samples.label_occurence,outs.occurence):
            occurence_pred=np.array(occ_out>0,dtype=np.float)
            accs += np.array(gt_value==occurence_pred,dtype=np.int)
            lens += 1
    #        for j,(gt_val, occ_pred) in enumerate(zip(gt_value,occurence_pred)):
    #            acc = int(gt_val==occ_pred)
    #            accs[j]+=acc
    #            lens[j]+=1

    print(accs/lens)
    # %%
    ro_labels=[]
    for inputs in train_dataset:
        inputs=tonp(inputs)
        samples = inputs_to_struct(inputs)
        flags = samples.flag
        label_tasks = samples.label_task
        for k in range(len(flags)):
            flag = flags[k]
            adj_type, char = flag_to_comp(flag)
            label_task = label_tasks[k][0]
            ro_labels.append(np.array([adj_type, char,label_task]))

    ro_labels=np.vstack(ro_labels)
    # %%
    inputs=next(iter(train_dataset));
    samples = inputs_to_struct(inputs);
    outs=model(inputs);
    outs = model.module.outs_to_struct(outs)
    ns_detach_tonp(outs)
    # %% store readouts
    load_model(train_opts,os.path.join(model_dir,'model_latest.pt'))
    model.eval()

    def store_sample_disk(sample,cur_samples_dir,split,splitsize):
        #os.makedirs(cur_samples_dir)
        samples_dir = cur_samples_dir
        i = sample.id
        if split:
            samples_dir= os.path.join(cur_samples_dir,'%d' % (i//splitsize))
            if not os.path.exists(samples_dir):
                os.makedirs(samples_dir,exist_ok=True)
        img = sample.image.transpose(1,2,0).astype(np.uint8)
        img_fname = os.path.join(samples_dir,'%d_img.jpg' % i)
        c = Image.fromarray(img)
        c.save(img_fname)
        seg = sample.seg.transpose(1,2,0).astype(np.uint8)
        img_fname = os.path.join(samples_dir,'%d_seg.jpg' % i)
        c = Image.fromarray(img)
        c.save(img_fname)
        del sample.image
        del sample.seg
        data_fname=os.path.join(samples_dir,'%d_raw.pkl' % i)
        with open(data_fname, "wb") as new_data_file:
            pickle.dump(sample, new_data_file)

    split = True
    splitsize = 1000
    flag_to_origsize=False
    from PIL import Image
    import skimage.transform
    def store_samples(ds,cur_samples_dir,is_training):
        i=0
        for inputs in ds:
            with torch.no_grad():
                outs = model(inputs)
                outs = model.module.outs_to_struct(outs)
                ns_detach_tonp(outs)

            inputs = tonp(inputs)
            cur_batch_size = inputs[0].shape[0] # last batch could be smaller than model_opts.batch_size
            for index in range(cur_batch_size):
                single_input=tuple(tensor[index] for tensor in inputs)
                sample = inputs_to_struct(single_input)
                sample.bu = outs.bu[index]
                sample.bu2 = outs.bu2[index]

                store_sample_disk(sample,cur_samples_dir,split,splitsize)
                i+=1
                if i % 1000 == 0:
                    print('%0.f %%' % (100*i/len(ds.dl.dataset)))
        print('done dataset')
        return i

    cur_samples_dir = os.path.join(model_dir,'samples')
    new_nsamples_train= store_samples(train_dataset,os.path.join(cur_samples_dir,'train'),True)
    new_nsamples_test = store_samples(test_dataset,os.path.join(cur_samples_dir,'test'),False)
    store_samples(val_dataset,os.path.join(cur_samples_dir,'val'),False)
    data_fname = os.path.join(cur_samples_dir,'conf')
    with open(data_fname, "wb") as new_data_file:
            pickle.dump((nsamples_train, nsamples_test, nsamples_val, nclasses_existence, img_channels, LETTER_SIZE, IMAGE_SIZE, ntypes, edge_class, not_available_class, total_rows, obj_per_row, sample_nchars, ngenerate, ndirections, exclude_percentage,valid_classes))

        # from shutil import copyfile
        # out_data_fname = os.path.join(out_tf_records_dir, 'conf')
        # copyfile(data_fname, out_data_fname)
