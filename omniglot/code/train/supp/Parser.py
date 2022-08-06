
from supp.FlagAt import *
from datetime import datetime
from supp.general_functions import create_dict
from supp.loss_and_accuracy import *
from supp.batch_norm import *
import torch.nn as nn
import argparse
from supp.blocks import *
from supp.create_model import create_model
from supp.omniglot_dataset import inputs_to_struct as inputs_to_struct
from supp.omniglot_dataset import get_omniglot_dictionary
from typing import Callable

def GetParser(model_flag, raw_data_path, processed_data, embedding_idx, checkpoints_per_epoch=1,stages = [0]):
    """
    Args:
        model_flag:
        raw_data_path:
        processed_data:
        embedding_idx:
        checkpoints_per_epoch:
        stages:

    Returns:
    """
    parser = argparse.ArgumentParser()
    num_gpus = torch.cuda.device_count()
    parser.add_argument('--stages', default=stages, type=list, help = "'")
    parser.add_argument('--grit_size', default = 6, type=list, help="'")
    parser.add_argument('--wd', default=0.00001, type=float, help='The weight decay of the Adam optimizer')
    parser.add_argument('--SGD', default=False, type=bool, help='Whether to use SGD or Adam optimizer')
    parser.add_argument('--lr', default=1e-3 * 2, type=float, help='Base lr for the SGD optimizer ')
    parser.add_argument('--checkpoints_per_epoch', default=checkpoints_per_epoch, type=int,  help='Number of model saves per epoch')
    parser.add_argument('--initial_tasks', default=[18, 42, 5, 27], type=list,  help='The initial tasks to train first')
    parser.add_argument('--bs', default=10, type=int, help='The training batch size')
    parser.add_argument('--scale_batch_size', default=num_gpus * parser.parse_args().bs, type=int,     help='scale batch size')
    parser.add_argument('--gpu', default=None, type=any, help='Not clear - query!')  # TODO-understand what is gpu.
    parser.add_argument('--distributed', default=False, type=bool,   help='Whether to use distributed data')  # TODO-understand what is distributed.
    parser.add_argument('--multiprocessing_distributed', default=False, type=bool,  help='Whether to use multiprocessing_distributed data')  # TODO-understand what it is.
    parser.add_argument('--workers', default=2, type=int, help='Number of workers to use')
    parser.add_argument('--avg_grad', default=False, type=bool,   help='Whether to average the gradient by the batch size')
    parser.add_argument('--cycle_lr', default=True, type=bool, help='Whether to cyclr the lr')
    parser.add_argument('--normalize_image', default=False, type=bool,  help='Whether to normalize the data by subtracting  the mean')
    parser.add_argument('--orig_relus', default=False, type=bool, help='Flag whether to add ReLu in certain places')
    parser.add_argument('--base_lr', default=0.0002, type=float, help='Base lr of the cyclic Adam optimizer')
    parser.add_argument('--max_lr', default=0.002, type=float,    help='Max lr of the cyclic Adam optimizer before the lr returns to the base_lr')
    parser.add_argument('--momentum ', default=0.9, type=float, help='Momentum of the Adam optimizer')
    parser.add_argument('--model_flag', default=model_flag, type=FlagAt, help='Flag that defines the model type')
    parser.add_argument('--ntasks', default=51, type=int, help='Number of tasks the model should handle')
    parser.add_argument('--nargs', default=6, type=int, help='Number of possible position arguments in the image')
    parser.add_argument('--activation_fun', default=nn.ReLU, type=nn.Module, help='Non Linear activation function')
    parser.add_argument('--use_bu1_loss', default=False, type=bool,   help='Whether to use the binary classification loss at the end of the BU1 stream')
    parser.add_argument('--use_td_loss', default=False, type=bool,     help='Whether to use the segmentation loss at the end of the TD stream')
    parser.add_argument('--use_bu2_loss', default=True, type=bool,       help='Whether to use the classification loss at the end of the BU2 stream')
    parser.add_argument('--use_lateral_butd', default=True, type=bool,      help='Whether to use the lateral connection from bu1 to TD')
    parser.add_argument('--use_lateral_tdbu', default=True, type=bool,       help='Whether to use the lateral connection from TD to BU2')
    parser.add_argument('--use_final_conv', default=False, type=bool,     help='Whether to use the final conc at the end of BU2')
    parser.add_argument('--nfilters', default = [64, 128, 256, 516], type=list, help='The ResNet filters')
    parser.add_argument('--strides', default=[2,2,2,2], type=list, help='The ResNet strides')
    parser.add_argument('--ks', default=[7, 3, 3,3], type=list, help='The kernel sizes')
    parser.add_argument('--ns', default=[2,2,2,2], type=list, help='Number of blocks per filter size')
    parser.add_argument('--inshape', default=(3, 112, 224), type=tuple, help='The input image shape')
    parser.add_argument('--norm_fun', default=BatchNorm, type=nn.Module, help='The used batch normalization')
    parser.add_argument('--EPOCHS', default=200, type=int, help='Number of epochs in the training')
    parser.add_argument('--num_gpus', default=num_gpus, type=int, help='number of used gpus')
    # Change to another function
    ##########################################
    now = datetime.now()
    dt_string = now.strftime("%d.%m.%Y%H:%M:%S")
    model_path = 'DS=' + str(processed_data) + "_embedding_idx=" + str(embedding_idx) + 'Stage='+str(stages) + "Time=" + str(dt_string)
    ##########################################
    parser.add_argument('--model_dir', default = os.path.join('../data/results/',model_path), type=str, help='The direction the model will be stored')
    parser.add_argument('--resulrs_dir', default = '.../data/results', type=str, help='The direction the model will be stored')
    parser.add_argument('--data_dir', default = '/home/sverkip/data/BU-TD/omniglot/data/new_samples/test_new_100K_samples', type=str,  help='The direction the model will be stored')
    parser.add_argument('--nclasses', default = get_omniglot_dictionary(parser.parse_args(), raw_data_path), type=list,  help='The sizes of the Linear layers')
    parser.add_argument('--logfname', default='log.txt', type=str, help='The name of the log file')
    parser.add_argument('--bu2_loss', default=multi_label_loss, type=Callable, help='The bu2 classification loss')
    parser.add_argument('--task_accuracy', default=multi_label_accuracy_base, type=Callable,  help='The accuracy function')
    parser.add_argument('--ubs', default=1, type=int, help='The accuracy function')
    learning_rates_mult = np.ones(parser.parse_args().EPOCHS)
    #  learning_rates_mult = get_multi_gpu_learning_rate(learning_rates_mult,num_gpus, 1,1)
    # Change to general: get_multi_gpu_learning_rate(learning_rates_mult,num_gpus, Parser.scale_batch_size,Parser.ubs)
    if checkpoints_per_epoch > 1:
        learning_rates_mult = np.repeat(learning_rates_mult, heckpoints_per_epoch)
    setup_flag(parser)
    parser.add_argument('--inputs_to_struct', default=inputs_to_struct, type=object,  help='struct transforming the list of data into struct.')
    parser.add_argument('--learning_rates_mult', default=learning_rates_mult, type=list, help='The scaled learning rated')
    parser.add_argument('--save_model', default=True, type=bool, help='Whether to save the model on the disk')
    parser.add_argument('--td_block_type', default=BasicBlockTD, type=nn.Module, help='Basic TD block')
    parser.add_argument('--bu_block_type', default=BasicBlockBU, type=nn.Module, help='Basic BU block')
    parser.add_argument('--bu_shared_block_type', default=BasicBlockBUShared, type=nn.Module,    help='Basic shared BU block')
    parser.add_argument('--bu1_loss', default=nn.BCEWithLogitsLoss(reduction='mean').to(dev), type=nn.Module,  help='The loss used at the end of the bu1 stream')
    parser.add_argument('--td_loss', default=nn.MSELoss(reduction='mean').to(dev), type=nn.Module,   help='The loss used at the end of the td stream')
    parser.add_argument('--bu2_classification_loss', default=nn.CrossEntropyLoss(reduction='none').to(dev),    type=nn.Module, help='The loss used at the end of the bu2 stream')
    parser.add_argument('--model', default=create_model(parser.parse_args()), type=nn.Module,   help='The model we fit')
    parser.add_argument('--loss_fun', default=CYCLICUnifiedLossFun(parser.parse_args()), type=Callable, help='The unified loss function of all training')
    parser.add_argument('--epoch_save_idx', default = 'accuracy', type=str, help='The metric we update the best model according to(usually loss/accuracy)')
    parser.add_argument('--dataset_id', default='test', type=str, help='The dataset we update the best model according to(usually val/test)')
    return parser.parse_args()


def get_multi_gpu_learning_rate(learning_rates_mult, num_gpus, scale_batch_size, ubs):
    # In pytorch gradients are summed across multi-GPUs (and not averaged) so
    # there is no need to change the learning rate when changing from a single GPU to multiple GPUs.
    # However when increasing the batch size (not because of multi-GPU, i.e. when scale_batch_size>1),
    # we need to increase the learning rate as usual
    clr = False
    if clr:
        learning_rates_mult *= scale_batch_size
    else:
        if ubs > 1:
            warmup_epochs = 5
            initial_lr = np.linspace(
                learning_rates_mult[0] / num_gpus, scale_batch_size * learning_rates_mult[0], warmup_epochs)
            learning_rates_mult = np.concatenate((initial_lr, scale_batch_size * learning_rates_mult))
    return learning_rates_mult
