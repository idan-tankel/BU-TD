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
            learned_params.extend(model.module.transfer_learning[lang_id])
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

def main_emnist(language_idx,direction_idx,train_right,train_left,wd):
    opts = Model_Options_By_Flag_And_DsType(Flag = Flag.SF, DsType = DsType.Emnist)
    path = "Model_right_wd="+str(wd)
    parser = GetParser(opts = opts, language_idx = language_idx,wd=wd,model_path=path)
    print_detail(parser)

    data_path = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/samples/18_extended_testing_new_changes_beta_emnist'
    # Create the data for right.
    [the_datasets, train_dl, test_dl ,val_dl,_, _, _] = get_dataset_for_spatial_realtions(parser, data_path, lan_idx=0, direction_idx=0,arg_idx=0)
    # Training Right.

    path_loading = 'Model_right_wd=1e-0521.09.2022 13:45:01/model_best_right.pt'
    model_path = parser.results_dir
    load_model(parser.model, model_path, path_loading, load_optimizer_and_schedular=False);
   # acc = accuracy(parser.model, test_dl)
   # print("Done training right, with accuracy : " + str(acc))

    if train_right:
        parser.EPOCHS = 40
        training_flag = Training_flag(train_all_model = True, train_arg=True,direction_emb = False,lang_emb = True, head_learning=True)
        train_omniglot(parser, lang_id = 0,direction_id=direction_idx, the_datasets=the_datasets, training_flag=training_flag)
   # reg = Regulizer(parser.reg, parser, test_dl)

   # parser.reg = reg
   # parser.use_reg = True
    if train_left:
        parser.EPOCHS = 100
        [the_datasets, _, _,test_dl,_, _, _] = get_dataset_for_spatial_realtions(parser, data_path, lan_idx=2, direction_idx = 2, arg_idx=0)

        training_flag = Training_flag(train_all_model=False, train_arg=False, direction_emb=False, lang_emb=True, head_learning=True)
        train_omniglot(parser, lang_id=2, direction_id = 0, the_datasets=the_datasets, training_flag=training_flag)
   # print("Done training left, with accuracy : " + str(acc))



main_emnist(0,0,False,True,wd=1e-5)