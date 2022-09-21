import os
#import argparse
import torch.backends.cudnn as cudnn
from supp.Parser import GetParser
from supp.get_dataset import get_dataset_for_spatial_realtions, get_dataset_cifar
from supp.FlagAt import Flag, DsType,Model_Options_By_Flag_And_DsType
from supp.logger import print_detail
from supp.loss_and_accuracy import accuracy
from supp.measurments import set_datasets_measurements
from supp.training_functions import load_model, create_optimizer_and_sched, train_model
from supp.general_functions import num_params
from supp.measurments import Measurements
from supp.batch_norm import load_running_stats
from supp.data_functions import dev
from supp.reg import Regulizer

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Training_flag:
    def __init__(self,train_all_model, train_arg,lang_emb,direction_emb,head_learning):
        self.train_all_model = train_all_model
        self.train_arg = train_arg
        self.lang_emb = lang_emb
        self.direction_emb = direction_emb
        self.head_learning = head_learning

    def Get_learned_params(self,model, lang_id, direction):
        learned_params = []
        if self.direction_emb:
            learned_params.extend(model.module.direction_embedding[direction])
        if self.lang_emb:
            learned_params.extend(model.module.lang_embedding[lang_id])
        if self.head_learning:
            learned_params.extend(list(model.module.Head.taskhead[lang_id][direction].parameters()))
        if self.train_arg:
            learned_params.extend(model.module.tdmodel.argument_embedding[lang_id])
        if self.train_all_model:
            learned_params = list(model.parameters())
    #    print(num_params(learned_params))
        return learned_params

def train_omniglot(parser, direction_id,lang_id, the_datasets, training_flag):
    set_datasets_measurements(the_datasets, Measurements, parser, parser.model)
    cudnn.benchmark = True  # TODO:understand what it is.
    # Deciding which parameters will be trained: if True all the model otherwise,only the task embedding.
    learned_params =training_flag.Get_learned_params(parser.model,lang_id,direction_id)
    # Training the learned params of the model.
    create_optimizer_and_sched(parser,learned_params)
    train_model(parser, the_datasets, learned_params, direction_id = direction_id , lang_id = lang_id)

def main_omniglot(language_idx,train_right,train_left,wd):
    # TODO - ARRANGE THIS MAIN FUCNTION, ON MONDAY!
    """
    Args:
        language_idx:
        train_right:
        train_left:
    Returns:
    """
    model_path ='Model_testing' +str(language_idx) + '_wd='+str(wd)
    opts = Model_Options_By_Flag_And_DsType(Flag=Flag.SF, DsType=DsType.Omniglot)
    parser = GetParser(opts=opts,model_path = model_path, language_idx=0,wd = wd)
    print_detail(parser)
    embedding_idx = 0
    data_path = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/omniglot/samples/6_extended_testing_new_changes_beta_5R'
    # Create the data for right.
    path_loading = 'Model_testing-1_wd=1e-05/model_best_right.pt'
    model_path = parser.results_dir
    load_model(parser.model, model_path, path_loading, load_optimizer_and_schedular=False);

    if train_right:
        parser.EPOCHS = 20
        [the_datasets, _, test_dl, _, _, _, _] = get_dataset_for_spatial_realtions(parser, data_path, direction_idx=0, arg_idx = language_idx,lan_idx = 0)
        training_flag = Training_flag(train_all_model = False, train_arg=True,direction_emb = False,lang_emb = False, head_learning=True)
        print(accuracy(parser.model, test_dl))
        train_omniglot(parser, direction_id = 0, the_datasets=the_datasets, training_flag=training_flag, lang_id = language_idx+1)

    if train_left:
        parser.EPOCHS = 100
        [the_datasets, _, _, _, _, _, _] = get_dataset_for_spatial_realtions(parser, data_path, direction_idx=1, arg_idx = language_idx, lan_idx = 0 )
        training_flag = Training_flag(train_all_model = False, train_arg = False, direction_emb = False,lang_emb = True, head_learning=True)
        train_omniglot(parser, direction_id = 1, the_datasets=the_datasets, training_flag=training_flag, lang_id = 0)

main_omniglot(-1,True, True,wd=1e-5)
#main_emnist(0,0,False,True,wd=1e-4)


#  path_loading = 'Model5R_wd=1e-05/model_best_right.pt'
# model_path = parser.results_dir
#  load_model(parser.model, model_path, path_loading, load_optimizer_and_schedular=False);

