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
    new_ds_conf=True
    emnist_dir = os.path.join(data_dir, 'emnist')
    if new_ds_conf:
        base_tf_records_dir = '6_extended50k1'
    else:
        base_tf_records_dir = 'emnist_adj_2dir_6_ids_10k_aug_genmore1nb_5cs_split'
    new_emnist_dir=os.path.join(emnist_dir,'samples')
    base_samples_dir=os.path.join(home_dir,"data/6_extended_testing")
    data_fname=os.path.join(base_samples_dir,'conf')
    results_dir = os.path.join(emnist_dir, 'results')
    flag_at=FlagAt.NOFLAG
    # when True use a dummy dataset instead of a real one (for debugging)
    dummyds =False
    cycle_lr = False

    update(locals())

def get_args(config=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--hyper', default=-1, type=int)
    parser.add_argument('--only_cont', action='store_true')
    parser.add_argument('--SGD', action='store_true')
    parser.add_argument('-lr', default=1e-3, type=float)
    parser.add_argument('-bs', default=192, type=int)
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
    dummyds = True
    if not dummyds:
        with open(data_fname, "rb") as new_data_file:
            if new_ds_conf:
                nsamples_train, nsamples_test, nsamples_val, nclasses_existence, img_channels, LETTER_SIZE, IMAGE_SIZE, ntypes, edge_class, not_available_class, total_rows, obj_per_row, sample_nchars, ngenerate, ndirections, exclude_percentage,valid_classes,cl2let = pickle.load(new_data_file)
                nclasses = int(max(ntypes))
            else:
                nsamples_train, nsamples_test, est_epochs, nclasses_existence, img_channels, total_bins, LETTER_SIZE, IMAGE_SIZE, ntypes, edge_class, not_available_class, total_rows,obj_per_row, sample_npersons, ngenerate, ndirections = pickle.load(
                    new_data_file)
                nclasses = int(max(ntypes))
                ntypes = nclasses * np.ones(nclasses_existence,dtype=np.int)
                # class to letter dictionary
                text_file = open("emnist-balanced-mapping.txt", "r")
                lines = text_file.read().split('\n')
                cl2letmap = [line.split() for line in lines]
                cl2letmap = cl2letmap[:-1]
                cl2let = {int(mapi[0]): chr(int(mapi[1])) for mapi in cl2letmap}
                cl2let[47] = 'Border'
                cl2let[not_available_class] = 'NA'
    else:
        nclasses_existence = 47
        not_available_class = nclasses_existence + 1
        nclasses = not_available_class + 1
        ndirections=2
        nsamples_train=200
        nsamples_test=200
        ntypes = nclasses * np.ones(nclasses_existence,dtype=np.int)
        IMAGE_SIZE = [224,224]
        img_channels = 3

    not_existing_class = not_available_class

    update(locals())

if interactive_session or is_main:
    load_samples()
# %% dataset
def dataset():
    inshape = (img_channels,*IMAGE_SIZE)
    flag_size=ndirections + nclasses_existence

    num_gpus = ngpus_per_node
    batch_size = args.bs
    scale_batch_size = 1
    ubs = scale_batch_size # unified batch scale
    if num_gpus > 1:
        ubs = ubs * num_gpus

    from torch.utils.data import DataLoader
    if dummyds:
        if flag_at is FlagAt.NOFLAG:
            from v27.emnist_dataset import EMNISTAdjDatasetDummyAdjAll as dataset, inputs_to_struct_label_adj_all as inputs_to_struct
        else:
            from v27.emnist_dataset import EMNISTAdjDatasetDummy as dataset, inputs_to_struct_basic as inputs_to_struct


        def flag_to_comp(flag):
            return 1, 1
        train_ds = dataset(inshape,nclasses_existence,ndirections,nsamples_train)
        test_ds = dataset(inshape,nclasses_existence,ndirections,nsamples_test)
        val_ds = dataset(inshape,nclasses_existence,ndirections,nsamples_test)
        nsamples_val = len(val_ds) # validation set is only sometimes present so nsamples_val is not always available
        normalize_image = False
    else:
        if new_ds_conf:
            if flag_at is FlagAt.NOFLAG:
                from v27.emnist_dataset import EMNISTAdjDatasetLabelAdjAllNew as dataset, inputs_to_struct_label_adj_all as inputs_to_struct
            else:
                from v27.emnist_dataset import EMNISTAdjDatasetNew as dataset, inputs_to_struct_basic as inputs_to_struct
        else:
            if flag_at is FlagAt.NOFLAG:
                from v27.emnist_dataset import EMNISTAdjDatasetLabelAdjAll as dataset, inputs_to_struct_label_adj_all as inputs_to_struct
            else:
                from v27.emnist_dataset import EMNISTAdjDataset as dataset, inputs_to_struct_basic as inputs_to_struct

        from v27.dataset_storage import Cache, CachedDataset
        use_cache = True
        def flag_to_comp(flag):
            adj_type_ohe = flag[:ndirections]
            adj_type = adj_type_ohe.nonzero()[0][0]
            char_ohe = flag[ndirections:]
            char = char_ohe.nonzero()[0][0]
            return adj_type, char

        train_ds = CachedDataset(dataset(os.path.join(base_samples_dir,'train'),47, 1, nexamples = nsamples_train,split = True), cache_supplier=Cache(shuffle=True,shuffle_type = args.distributed,num_gpus=num_gpus) if use_cache else None)
        normalize_image = False
        if normalize_image:
            # just for getting the mean image
            train_dl = DataLoader(train_ds, batch_size=batch_size,num_workers=args.workers,shuffle=False,pin_memory=True)
            mean_image = retrieve_mean_image(train_dl,inshape,inputs_to_struct, base_samples_dir, True)
            train_ds = CachedDataset(dataset(os.path.join(base_samples_dir,'train'),nclasses_existence, ndirections, nexamples = nsamples_train,split = True,mean_image = mean_image), cache_supplier=Cache(shuffle=True,shuffle_type = args.distributed,num_gpus=num_gpus) if use_cache else None)
        else:
            mean_image = None
        test_ds = CachedDataset(dataset(os.path.join(base_samples_dir,'test'),nclasses_existence, ndirections, nexamples = nsamples_test,split = True,mean_image = mean_image), cache_supplier=Cache() if use_cache else None)
        val_ds = CachedDataset(dataset(os.path.join(base_samples_dir,'val'),nclasses_existence, ndirections, nexamples = nsamples_test, split = True,mean_image = mean_image), cache_supplier=Cache() if use_cache else None)

        nsamples_val = len(val_ds) # validation set is only sometimes present so nsamples_val is not always available

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds,shuffle=not use_cache)
        if False:
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

    train_dl = DataLoader(train_ds, batch_size=batch_size,num_workers=args.workers,shuffle=True,pin_memory=True, sampler=train_sampler)
    # train_dl = DataLoader(train_ds, batch_size=batch_size,num_workers=0,shuffle=False,pin_memory=True, sampler=train_sampler)
    test_dl = DataLoader(test_ds, batch_size=batch_size,num_workers=args.workers,shuffle=False,pin_memory=True, sampler=test_sampler)
    val_dl = DataLoader(val_ds, batch_size=batch_size,num_workers=args.workers,shuffle=False,pin_memory=True, sampler=val_sampler)

    nbatches_train = len(train_dl)
    nbatches_val = len(val_dl)
    nbatches_test = len(test_dl)
    # commented out as in distributed, each train loader using the sampler will have the correct size: nsamples/(batch_size*ngpus)
    # nbatches_train = int(np.ceil(nsamples_train / batch_size))
    # nbatches_val = int(np.ceil(nsamples_test / batch_size))
    # nbatches_test = int(np.ceil(nsamples_test / batch_size))
    # print(batch_size,nbatches_train)

    train_dataset = WrappedDataLoader(train_dl, preprocess)
    test_dataset = WrappedDataLoader(test_dl, preprocess)
    val_dataset = WrappedDataLoader(val_dl, preprocess)

    the_train_dataset = DatasetInfo(True,train_dataset,nbatches_train,'Train',args.checkpoints_per_epoch,train_sampler)
    the_test_dataset = DatasetInfo(False,test_dataset,nbatches_test,'Test',1,test_sampler)
    the_datasets = [the_train_dataset,the_test_dataset]
    # the_datasets = [the_train_dataset]
    if nsamples_val>0:
        the_val_dataset = DatasetInfo(False,val_dataset,nbatches_val,'Validation',1,val_sampler)
        the_datasets += [the_val_dataset]
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
    model_opts.normalize_image = normalize_image
    model_opts.flag_at=flag_at
    model_opts.noflag_cs = False
    model_opts.nclasses_existence = nclasses_existence
    if model_opts.flag_at is FlagAt.NOFLAG:
        model_opts.nclasses = ntypes
    else:
        model_opts.nclasses = [nclasses]

    model_opts.flag_size=flag_size
    model_opts.activation_fun = nn.ReLU
    model_opts.use_td_loss = False
    model_opts.use_bu1_loss = True
    model_opts.use_bu2_loss = True
    model_opts.use_lateral_butd = True
    model_opts.use_lateral_tdbu = True
    model_opts.ntaskhead_fc = 1
    model_opts.cycle_lr = cycle_lr
    model_opts.nfilters=[128]
    model_opts.inshape=inshape
    use_sepnorm = False
    model_opts.use_sepnorm = use_sepnorm
    setup_flag(model_opts)

    num_gpus = torch.cuda.device_count()
    dummytype = '_dummyds' * dummyds
    cyclelr_st = '_cyclr' * cycle_lr
    opt_params_st ='ray_' * optimize_hyperparams
    flag_str = str(model_opts.flag_at).replace('.','_').lower()
    flag_str = flag_str + 'cs' * model_opts.noflag_cs
    base_model_dir = 'emnist_pyt_%sv27_vit_cct2_%s_prefood' % (opt_params_st,flag_str)
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
    logger.info(args)
    update(locals())

if interactive_session or is_main:
    set_model_opts()
# %% create model
import torch
import torch.nn as nn
import torch.nn.functional as F
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# modules

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, inputs):
        x, q_hat, k_hat, v_hat = inputs
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if q_hat is not None:
            q = q + q_hat
            k = k + k_hat
            v = v + v_hat

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, q, k, v


class TransformerEncoderLayerSharedBase():
    """
    Inspired by torch.nn.TransformerEncoderLayer and
    rwightman's timm package.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1):
        super(TransformerEncoderLayerSharedBase, self).__init__()
        self.d_model = d_model
        if not use_sepnorm:
            self.pre_norm = nn.LayerNorm(d_model)
            self.norm1    = nn.LayerNorm(d_model)
        self.self_attn = Attention(dim=d_model, num_heads=nhead,
                                   attention_dropout=attention_dropout, projection_dropout=dropout)

        self.linear1  = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2  = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        self.activation = F.gelu

class TransformerEncoderLayerShared(nn.Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and
    rwightman's timm package.
    """
    def __init__(self, shared):
        super(TransformerEncoderLayerShared, self).__init__()
        if use_sepnorm:
            self.pre_norm = nn.LayerNorm(shared.d_model)
        else:
            self.pre_norm = shared.pre_norm
        self.self_attn = shared.self_attn

        self.linear1  = shared.linear1
        self.dropout1 = shared.dropout1
        if use_sepnorm:
            self.norm1    = nn.LayerNorm(shared.d_model)
        else:
            self.norm1    = shared.norm1
        self.linear2  = shared.linear2
        self.dropout2 = shared.dropout2

        self.drop_path = shared.drop_path

        self.activation = shared.activation

    def forward(self, inputs, *args, **kwargs) -> torch.Tensor:
        x, q_hat, k_hat, v_hat = inputs
        x_attn, q, k, v = self.self_attn((self.pre_norm(x), q_hat, k_hat, v_hat))
        src = x + self.drop_path(x_attn)        
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src, q, k, v


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Tokenizer(nn.Module):
    def __init__(self,
                 kernel_size, stride, padding,
                 pooling_kernel_size=3, pooling_stride=2, pooling_padding=1,
                 n_conv_layers=1,
                 n_input_channels=3,
                 n_output_channels=64,
                 in_planes=64,
                 activation=None,
                 max_pool=True,
                 conv_bias=False,
                 ):
        super(Tokenizer, self).__init__()

        n_filter_list = [n_input_channels] + \
                        [in_planes for _ in range(n_conv_layers - 1)] + \
                        [n_output_channels]

        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(n_filter_list[i], n_filter_list[i + 1],
                          kernel_size=(kernel_size, kernel_size),
                          stride=(stride, stride),
                          padding=(padding, padding), bias=conv_bias),
                nn.Identity() if activation is None else activation(),
                nn.MaxPool2d(kernel_size=pooling_kernel_size,
                             stride=pooling_stride,
                             padding=pooling_padding) if max_pool else nn.Identity()
            )
                for i in range(n_conv_layers)
            ])

        self.flattener = nn.Flatten(2, 3)
        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def forward(self, x):
        return self.flattener(self.conv_layers(x)).transpose(-2, -1)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class PositionalEmbedding(nn.Module):
    def __init__(self,
                 seq_pool,
                 embedding_dim,
                 positional_embedding,
                 sequence_length,
                 dropout_rate):
        super().__init__()
        self.seq_pool = seq_pool
        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'

        assert sequence_length is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not seq_pool:
            sequence_length += 1
            self.class_emb = nn.Parameter(torch.zeros(1, 1, embedding_dim),
                                          requires_grad=True)

        self.sequence_length = sequence_length
        if positional_embedding != 'none':
            if positional_embedding == 'learnable':
                self.positional_emb = nn.Parameter(torch.zeros(1, sequence_length, embedding_dim),
                                                   requires_grad=True)
                nn.init.trunc_normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = nn.Parameter(self.sinusoidal_embedding(sequence_length, embedding_dim),
                                                   requires_grad=False)
        else:
            self.positional_emb = None
        self.dropout = nn.Dropout(p=dropout_rate)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)

    def forward(self, x):
        if self.positional_emb is None and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)

        if not self.seq_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        if self.positional_emb is not None:
            x += self.positional_emb
        x = self.dropout(x)
        return x

class TransformerPool(nn.Module):
    def __init__(self,
                 seq_pool=True,
                 embedding_dim=768,
                 *args, **kwargs):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.seq_pool = seq_pool

        if seq_pool:
            self.attention_pool = nn.Linear(self.embedding_dim, 1)

        self.norm = nn.LayerNorm(embedding_dim)

        self.apply(self.init_weight)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.norm(x)

        if self.seq_pool:
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        else:
            x = x[:, 0]
        # after pool x size is 1 x embedding_dim
        
        return x

class TransformerEncoderSharedBase(nn.Module):
    def __init__(self,
                 embedding_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 dropout_rate=0.1,
                 attention_dropout=0.1,
                 stochastic_depth_rate=0.1,
                 *args, **kwargs):
        super().__init__()
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, num_layers)]
        self.blocks = [
            TransformerEncoderLayerSharedBase(d_model=embedding_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward, dropout=dropout_rate,
                                    attention_dropout=attention_dropout, drop_path_rate=dpr[i])
            for i in range(num_layers)]

        self.apply(self.init_weight)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def get_laterals(laterals_in,lateral_layer_id):
    return laterals_in[lateral_layer_id]

class TransformerEncoderShared(nn.Module):
    def __init__(self, shared, lateral_active, opts):
        super(TransformerEncoderShared, self).__init__()
        self.use_lateral = lateral_active #incoming lateral
        self.blocks = nn.ModuleList([TransformerEncoderLayerShared(shared_block) for shared_block in shared.blocks])
        
    def forward(self, inputs):
        x, laterals_in = inputs

        laterals_out = []
        # laterals_out.append(x)

        for layer_id,blk in enumerate(self.blocks):
            lateral_layer_id = layer_id
            if self.use_lateral:
                x_lat_in, q_lat_in, k_lat_in, v_lat_in = get_laterals(laterals_in,lateral_layer_id)
                x = x + x_lat_in
            else:
                q_lat_in, k_lat_in, v_lat_in = None, None, None
            x, q, k, v = blk((x, q_lat_in, k_lat_in, v_lat_in))
            laterals_out.append((x, q, k, v))

        return x,laterals_out


class CCTLatSharedBase():
    def __init__(self,
                 opts,
                 img_size=224,
                 embedding_dim=768,
                 n_input_channels=3,
                 n_conv_layers=1,
                 kernel_size=7,
                 stride=2,
                 padding=3,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 pooling_padding=1,
                 seq_pool=True,
                 *args, **kwargs):
        super(CCTLatSharedBase, self).__init__()
        img_height, img_width = pair(img_size)
        self.embedding_dim = embedding_dim
        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   pooling_kernel_size=pooling_kernel_size,
                                   pooling_stride=pooling_stride,
                                   pooling_padding=pooling_padding,
                                   max_pool=True,
                                   activation=nn.ReLU,
                                   n_conv_layers=n_conv_layers,
                                   conv_bias=False)
        self.sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_height,
                                                           width=img_width)
        self.seq_pool=seq_pool
        dropout_rate=0.1
        self.positional_embedding = PositionalEmbedding(seq_pool=seq_pool,
                                                        embedding_dim=embedding_dim,
                                                        positional_embedding='sine',
                                                        sequence_length=self.sequence_length,
                                                        dropout_rate=dropout_rate)
        self.encoder = TransformerEncoderSharedBase(
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate,
            attention_dropout=0.1,
            stochastic_depth_rate=0,
            *args, **kwargs)

        self.pool = TransformerPool(
            seq_pool=seq_pool,
            embedding_dim=embedding_dim,
            *args, **kwargs)


class CCTLatShared(nn.Module):
    def __init__(self,
                 opts, shared, lateral_active, use_top_flag):
        super(CCTLatShared, self).__init__()
        self.use_lateral = lateral_active #incoming lateral
        self.tokenizer = shared.tokenizer
        self.positional_embedding = shared.positional_embedding
        self.encoder = TransformerEncoderShared(shared.encoder, lateral_active, opts)
        self.pool = shared.pool
        self.activation_fun = opts.activation_fun
        self.sequence_length = shared.sequence_length
        self.embedding_dim = shared.embedding_dim
        self.use_top_flag = use_top_flag
        if self.use_top_flag:
            self.h_flag_bu2 = nn.Sequential(nn.Linear(opts.flag_size,self.embedding_dim),nn.LayerNorm(self.embedding_dim),self.activation_fun())
            self.h_top_bu2 = nn.Sequential(nn.Linear(self.embedding_dim*2,self.embedding_dim),nn.LayerNorm(self.embedding_dim),self.activation_fun())
        
    def forward(self, inputs):
        images, flags, laterals_in = inputs
        x = self.tokenizer(images)
        x = self.positional_embedding(x)
        model_inputs = [x, laterals_in]
        x,laterals_out = self.encoder(model_inputs)
        x = self.pool(x)
        if self.use_top_flag:
            flag_bu2 = self.h_flag_bu2(flags)
            x = torch.cat((x, flag_bu2),dim=1)
            x = self.h_top_bu2(x)

        return x,laterals_out


class CCTTDLat(nn.Module):

    def __init__(self, sequence_length, embedding_dim, num_layers, num_heads, mlp_ratio, seq_pool, opts):
        super(CCTTDLat, self).__init__()
        self.use_lateral = opts.use_lateral_butd
        self.use_td_flag = opts.use_td_flag
        self.activation_fun = opts.activation_fun
        self.sequence_length = sequence_length

        self.embedding_dim = embedding_dim
        if opts.use_td_flag:
            self.h_flag_td = nn.Sequential(nn.Linear(opts.flag_size,self.embedding_dim),nn.LayerNorm(embedding_dim),self.activation_fun())

        self.seq_pool = seq_pool
        if not self.seq_pool:
            self.class_emb = nn.Parameter(torch.zeros(1, 1, embedding_dim),
                                          requires_grad=True)
        self.encoder = TransformerEncoderSharedBase(
            embedding_dim=embedding_dim,
            dropout_rate=0.1,
            attention_dropout=0.1,
            stochastic_depth_rate=0,
            num_layers=num_layers,
            num_heads = num_heads,
            mlp_ratio = mlp_ratio)
        self.encoder = TransformerEncoderShared(self.encoder, self.use_lateral, opts)

        self.apply(self.init_weight)
        # init_module_weights(self.modules())

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inputs):
        bu_out, flag, laterals_in = inputs

        if self.use_td_flag:
            top_td = self.h_flag_td(flag)
            top_td = top_td.view((-1,1,self.embedding_dim))
            top_td = top_td.repeat(1,self.sequence_length,1)
            x = top_td
        else:
            x = laterals_in[-1][0] # the output of BU1 before pooling 

        if not self.seq_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        if laterals_in is None or not self.use_lateral:
            #TODO: write
            pass
        else:
            reverse_laterals_in = laterals_in[::-1]

        model_inputs = [x, reverse_laterals_in]
        x,laterals_out = self.encoder(model_inputs)

        outs = [x,laterals_out[::-1]]
        return outs

class CCTBUTDModel(nn.Module):

    def forward(self, inputs):
        samples = self.inputs_to_struct(inputs)
        images = samples.image/255
        flags = samples.flag
        model_inputs = [images,flags, None]
        bu_out, bu_laterals_out = self.bumodel1(model_inputs)
        if self.use_bu1_loss:
            occurence_out = self.occhead(bu_out)
        else:
            occurence_out = None
        model_inputs = [bu_out, flags]
        if self.use_lateral_butd:
            model_inputs += [bu_laterals_out]
        else:
            model_inputs += [None]
        td_outs = self.tdmodel(model_inputs)
        td_out, td_laterals_out, *td_rest = td_outs
        # if self.use_td_loss:
        #     td_head_out = self.imagehead(td_out)
        model_inputs = [images, flags]
        if self.use_lateral_tdbu:
            model_inputs += [td_laterals_out]
        else:
            # when not using laterals we only use td_out as a lateral
            model_inputs += [[td_out]]
        bu2_out, bu2_laterals_out = self.bumodel2(model_inputs)
        task_out = self.taskhead(bu2_out)
        outs = [occurence_out, task_out, bu_out, bu2_out]
        if self.use_td_loss:
            outs+= [td_head_out]
        # if self.tdmodel.trunk.use_td_flag:
        #     td_top_embed, td_top = td_rest
        #     outs+= [td_top_embed]
        return outs

    def outs_to_struct(self,outs):
        occurence_out, task_out, bu_out, bu2_out, *rest = outs
        outs_ns = SimpleNamespace(occurence=occurence_out,task=task_out,bu=bu_out,bu2=bu2_out)
        if self.use_td_loss:
            td_head_out, *rest = rest
            outs_ns.td_head = td_head_out
        # if self.tdmodel.trunk.use_td_flag:
        #     td_top_embed = rest[0]
        #     outs_ns.td_top_embed = td_top_embed
        return outs_ns

class CCTBUTDModelShared(CCTBUTDModel):

    def __init__(self, opts):
        super(CCTBUTDModelShared, self).__init__()
        self.use_bu1_loss = opts.use_bu1_loss
        if self.use_bu1_loss:
            self.occhead = OccurenceHead(opts)
        self.taskhead = MultiLabelHead(opts)
        self.use_td_loss = opts.use_td_loss
        # if self.use_td_loss:
        #     self.imagehead=ImageHead(opts)
        seq_pool = True
        bu_shared = CCTLatSharedBase(opts,
                        img_size = inshape[1:],
                        embedding_dim = opts.nfilters[-1],
                        n_conv_layers = 2,
                        kernel_size = 3,
                        stride = 2,
                        padding = 3,
                        pooling_kernel_size = 3,
                        pooling_stride = 2,
                        pooling_padding = 1,
                        num_layers = 2,
                        num_heads = 2,
                        mlp_ratio = 1.,
                        seq_pool = seq_pool,
                        )
        self.bumodel1=CCTLatShared(opts,bu_shared,lateral_active=False,use_top_flag=False)
        self.bumodel2=CCTLatShared(opts,bu_shared,lateral_active=True,use_top_flag=opts.use_bu2_flag)

        self.tdmodel=CCTTDLat(bu_shared.sequence_length, bu_shared.embedding_dim, bu_shared.encoder.num_layers, bu_shared.encoder.num_heads, bu_shared.encoder.mlp_ratio, seq_pool, opts)
        # self.use_bu1_flag = opts.use_bu1_flag
        self.use_lateral_butd = opts.use_lateral_butd
        self.use_lateral_tdbu = opts.use_lateral_tdbu
        self.inputs_to_struct = opts.inputs_to_struct


class BUModelClassification(nn.Module):

    def __init__(self, cctmodel, opts):
        super(BUModelClassification, self).__init__()
        v = cctmodel(
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
            num_layers = 2,
            num_heads = 2,
            mlp_ratio = 1.,
        )           

        self.bumodel=v
        self.use_bu1_loss = opts.use_bu1_loss
        self.use_bu2_loss = opts.use_bu2_loss
        if self.use_bu1_loss:
            self.occhead = OccurenceHead(opts)
        if self.use_bu2_loss:
            self.taskhead = MultiLabelHead(opts)

    def forward(self, inputs):
        samples = inputs_to_struct(inputs)
        model_inputs = samples.image

        bu_out = self.bumodel(model_inputs)
        if self.use_bu1_loss:
            occurence_out = self.occhead(bu_out)
        if self.use_bu2_loss:
            task_out = self.taskhead(bu_out)
        return occurence_out, task_out

    def outs_to_struct(self,outs):
        occurence_out,task_out = outs
        outs_ns = SimpleNamespace(occurence=occurence_out,task=task_out)
        return outs_ns
    

def create_model():
    if model_opts.flag_at is FlagAt.NOFLAG and not model_opts.noflag_cs:
        from vit_pytorch.cct import CCT        
        cctmodel = CCT
        model = BUModelClassification(cctmodel, model_opts)
    else:
        model = CCTBUTDModelShared(model_opts)

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

use_sparse_occ_loss = False
weighted_cross_entropy=False
if use_sparse_occ_loss and not weighted_cross_entropy:
    import torch.nn.functional as F
    class WeightedFocalLoss(nn.Module):
        "Non weighted version of Focal Loss"
        def __init__(self, alpha=.25, gamma=2):
            super(WeightedFocalLoss, self).__init__()
            self.alpha = torch.tensor([alpha, 1-alpha]).to(dev)
            self.gamma = gamma

        def forward(self, inputs, targets):
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            targets = targets.type(torch.long)
            at = self.alpha.gather(0, targets.data.view(-1)).view(targets.shape)
            pt = torch.exp(-BCE_loss)
            F_loss = at*(1-pt)**self.gamma * BCE_loss
            return F_loss.mean()

def loss_fun(inputs, outs):
    # nn.CrossEntropyLoss on GPU is not deterministic. However using CPU doesn't seem to help either...
    outs = get_model_outs(model,outs)
    samples = inputs_to_struct(inputs)
    losses=[]
    if model_opts.use_bu1_loss:
        if use_sparse_occ_loss:
            if weighted_cross_entropy:
                # balance the sparse occurence label between existing and non-existing. w.sum(axis=1) should be ~(62-8)*2
                le = samples.label_existence
                le_shape_prod = np.product(le.shape)
                le_exist = le.sum()
                w= (le_shape_prod-le_exist)/le_exist*le
                w[w==0]=1
                loss_occurence = torch.nn.BCEWithLogitsLoss(reduction='mean',pos_weight=w).to(dev)
            else:
                # loss_occ = torchvision.ops.sigmoid_focal_loss(outs.occurence,samples.label_existence,reduction='mean').to(dev)
                loss_occurence = WeightedFocalLoss().to(dev)
        else:
            loss_occurence = torch.nn.BCEWithLogitsLoss(reduction='mean').to(dev)

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
    # loss_occurence = torch.nn.BCEWithLogitsLoss(reduction='mean').to(dev)
    loss_seg = torch.nn.MSELoss(reduction='mean').to(dev)
    loss_task_op = nn.CrossEntropyLoss(reduction='mean').to(dev)
    loss_task_multi_label = nn.CrossEntropyLoss(reduction='none').to(dev)

    set_datasets_measurements(the_datasets,Measurements,model_opts,model)

    if model_opts.flag_at is FlagAt.NOFLAG:
        model_opts.bu2_loss = multi_label_loss_weighted_loss
        model_opts.task_accuracy = multi_label_accuracy_weighted_loss
    else:
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
    learning_rates_mult = np.ones(400)
    if args.avg_grad:
        learning_rates_mult = ubs *learning_rates_mult
    learning_rates_mult = get_multi_gpu_learning_rate(learning_rates_mult,num_gpus,scale_batch_size,ubs)
    if args.checkpoints_per_epoch>1:
        learning_rates_mult = np.repeat(learning_rates_mult,args.checkpoints_per_epoch)
    train_opts.batch_size = batch_size
    train_opts.nbatches_train = nbatches_train # just for logging
    train_opts.nbatches_val = nbatches_val # just for logging
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
        optimizer = optim.Adam(learned_params, lr = train_opts.initial_lr, weight_decay=train_opts.weight_decay)
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
    
    checkpoint = torch.load(os.path.join('/home/idanta/data/emnist/results/v27_vit_p16_normimg_nowd_cct4_224sz_128dim_food101','model_latest.pt'))
    state_dict = checkpoint['model_state_dict']
    for key in ['module.taskhead.layers.0.weight','module.taskhead.layers.0.bias']:
        state_dict[key]=model.state_dict()[key]
    model.load_state_dict(state_dict,strict=False)
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
        config = {"lr": tune.choice([0.0001,0.0002,0.0005,0.001]),
              'wd': tune.choice([0,0.0001,0.0005]),
              'bs': tune.choice([196]),
              'opt': tune.choice(['ADAM','AdaBelief']),
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
    scheduler = ASHAScheduler(metric=metric,mode=mode,max_t=100,grace_period=30,reduction_factor=2)

    if use_nevergrad:
        import nevergrad as ng
        from ray.tune.suggest.nevergrad import NevergradSearch
        algo = NevergradSearch(optimizer=ng.optimizers.OnePlusOne,metric=metric,mode=mode,space=space)
        from ray.tune.suggest import ConcurrencyLimiter
        algo = ConcurrencyLimiter(algo, max_concurrent=4)
    elif False:
        from ray.tune.suggest.hyperopt import HyperOptSearch
        algo = HyperOptSearch(metric=metric,mode=mode)
    else:
        from ray.tune.suggest.basic_variant import BasicVariantGenerator
        algo = BasicVariantGenerator()

    exp_name = args.exp
    ray_dir = '/home/liav/ray_results'
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

    reporter = CLIReporter2(max_report_frequency = 120,
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
            # resume = "ERRORED_ONLY"

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
    def from_network_transpose(samples,outs):
        if normalize_image:
            samples.image+=mean_image
        samples.image=samples.image.transpose(0,2,3,1)
        samples.seg=samples.seg.transpose(0,2,3,1)
        if model_opts.use_td_loss:
            outs.td_head=outs.td_head.transpose(0,2,3,1)
        return samples,outs


    ds_iter = iter(train_dataset)
    inputs=next(ds_iter)
    loss, outs = test_step(inputs,train_opts)
    samples,outs = from_network(inputs,outs,model.module,inputs_to_struct)
    samples,outs = from_network_transpose(samples,outs)
    preds=np.array(outs.occurence>0,dtype=np.float)
    fig = plt.figure(figsize=(15, 4))
    if model_opts.use_td_loss:
        n=3
    else:
        n=2
    for k in range(len(samples.image)):
        fig.clf()
        fig.tight_layout()
        ax=plt.subplot(1, n, 1)
        ax.axis('off')
        plt.imshow(samples.image[k].astype(np.uint8))
        # existence information without order
        present = samples.label_existence[k].nonzero()[0].tolist()
        present_st = [cl2let[c] for c in present]
        flag = samples.flag[k]
        if model_opts.flag_at is FlagAt.NOFLAG:
            tit = 'Right of all'
        else:
            adj_type, char = flag_to_comp(flag)
            adj_type_st = 'Right' if adj_type==0 else 'Left'
            ins_st = '%s of %s' % (adj_type_st,cl2let[char])
            tit = 'Present: %s...\n Instruction: %s ' % (present_st[:10],ins_st)
        plt.title(tit)

        ax=plt.subplot(1, n, 2)
        ax.axis('off')

        if model_opts.flag_at is FlagAt.NOFLAG:
            pred = outs.task[k].argmax(axis=0)
            gt = samples.label_task[k]
            gt_st = [cl2let[let] for let in gt]
            pred_st =[cl2let[let] for let in pred]
            font = {'color':  'blue'}
            gt_str = 'Ground Truth:\n%s...' % gt_st[:10]
            pred_str = 'Prediction:\n%s...' % pred_st[:10]
            print(gt_st)
            print(pred_st)
        else:
            gt_val = samples.label_task[k][0]
            pred_val=outs.task[k].argmax()
            if gt_val == pred_val:
                font = {'color':  'blue'}
            else:
                font = {'color':  'red'}
            gt_str = 'Ground Truth: %s' % cl2let[gt_val]
            pred_str = 'Prediction: %s' % cl2let[pred_val]
            print(char,cl2let[char],gt_val,cl2let[gt_val],pred_val,cl2let[pred_val])
        if model_opts.use_td_loss:
            tit_str = gt_str
            plt.title(tit_str)
        else:
            tit_str = gt_str+'\n'+pred_str
            plt.title(tit_str,fontdict=font)
        plt.imshow(samples.image[k].astype(np.uint8))
        # imshow(samples.seg[k] / 255)
        if model_opts.use_td_loss:
            ax=plt.subplot(1, n, 3)
            ax.axis('off')
            image_tdk = np.array(outs.td_head[k])
            image_tdk = image_tdk - np.min(image_tdk)
            image_tdk = image_tdk / np.max(image_tdk)
            plt.imshow(image_tdk)
            plt.title(pred_str,fontdict=font)
        print(k)
        label_all = samples.label_all[k]
        print(label_all)
        label_all_chars = [[cl2let[c] for c in row] for row in label_all]
        print(label_all_chars)
        #    savefig(os.path.join(emnist_dir, 'examples%d.png'% k) )
        pause_image()
    # %% percent correct - task
    conf = np.zeros((nclasses, nclasses))
    for inputs in test_dataset:
        loss, outs = test_step(inputs,train_opts)
        samples,outs = from_network(inputs,outs,model.module,inputs_to_struct)
        for k in range(len(samples.image)):
            if model_opts.flag_at is FlagAt.NOFLAG:
                pred = outs.task[k].argmax(axis=0)
                gt = samples.label_task[k]
                flag = samples.flag[k]
                adj_type, char = flag_to_comp(flag)
                if False:
                    # accuracy for all the characters
                    for gt_val,pred_val in zip (gt,pred):
                        if gt_val!=not_existing_class:
                            # if gt_val!=edge_class:
                                conf[gt_val][pred_val]+=1
                else:
                    # accuracy only for the flag
                    if adj_type == 0:
                        gt_val = gt[char]
                        pred_val = pred[char]
                        conf[gt_val][pred_val]+=1
            else:
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
