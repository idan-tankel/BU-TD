import os
import argparse
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
# from utils.funcs import *
from supp.Parser import *
from supp.get_dataset import *
from supp.create_model import *
from supp.FlagAt import *
from supp.cifar_dataset import inputs_to_struct as inputs_to_struct
from supp.logger import *
from supp.data_functions import *
from supp.loss_and_accuracy import *
from supp.visuialize_predctions import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'



def train_cifar10(parser, embedding_idx, the_datasets):
    set_datasets_measurements(the_datasets, Measurements, parser, parser.model)
    cudnn.benchmark = True  # TODO:understand what it is.
    # Deciding which parameters will be trained: if True all the model otherwise,only the task embedding.
    learned_params = parser.model.parameters()
    # Training the learned params of the model.
    return train_model(parser, the_datasets, learned_params, embedding_idx)

def main(lr, wd, lr_decay):
    parser = GetParser(lr , wd, lr_decay )
    print_detail(parser)
    data_path = '/home/projects/shimon/sverkip/cifar10/data/processed'
    # Create the data for right.
    [the_datasets, _, test_dl, _, _] = get_dataset( parser, root = data_path)
    model_path = parser.results_dir
    return train_cifar10(parser,embedding_idx = 0,the_datasets = the_datasets)

def find_best_hyper_params():
    lrs = [0.01, 0.02,0.001,0.03,0.005,0.002]
    wds = [0.0,1e-5,1e-6,1e-4,1e-3,1e-2 ]
    lrs_decay = [1.0]
    optimum = 0.0
    index = [0.0, 0.0, 0.0]
    for lr in lrs:
     for wd in wds:
      for lr_decay in lrs_decay:
          save_details = main(lr,wd,lr_decay)   
          new_optimum = save_details.optimum
          if new_optimum > optimum:
              optimum = new_optimum
              index = [lr,wd,lr_decay]
    print("The best model with optimum: " + str(optimum))
    print("The hyper-params are : "+"lr = "+str(index[0]),"wd = "+str(index[1])+"lr_decay = "+str(index[2]))
                
find_best_hyper_params()

# 6_extended_[27, 5, 42, 18].
# Temp place
##############################################
# print(accuracy(parser, test_dl))
# visualize(parser, train_dataset)
# /home/sverkip/data/Omniglot/data/new_samples/T
# print(num_params(learned_params))
# print(accuracy_one_language(args.model, args.test_dl))
# /home/sverkip/data/Omniglot/data/results/DS=Four_languages_embedding_idx=029.06.2022 11:55:49/model5.pt
# 'DS=8_extended_exp[27, 5, 42]_embedding_idx=029.06.2022 15:21:58'
###############################################
