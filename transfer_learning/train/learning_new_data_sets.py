"""
Train classification data-sets continually.
"""
from src.data.Parser import GetParser, update_parser
from src.python_lightning.Wrapper import ModelWrapped
from pathlib import Path
import os
import torch
from src.Utils import num_params, Load_pretrained_model, Get_dataloaders, \
    Get_Model_path, Get_Learned_Params, Define_Trainer, load_model
from src.data.Enums import data_set_types, Model_type, TrainingFlag, ModelFlag
from src.Modules.Create_model import create_model


def train_classification(task_id: int, load: bool, ds_type: data_set_types,
                         training_flag: TrainingFlag, training_type: ModelFlag,
                         model_type=Model_type.ResNet18, load_pretrained_model: bool = True,
                         weight_modulation_factor=None) \
        -> None:
    """
    Train cifar continual learning.
    Args:
        task_id: The task id in [0,10]
        load: Whether to load pretrained model.
        ds_type: The data-set type.
        training_flag: The training flag.
        training_type: The model type.
        model_type: The model type,
        load_pretrained_model: Load pretrained model.
        weight_modulation_factor: The weight modulation factor.

    """
    opts = GetParser(ds_type=ds_type, model_type=model_type, ModelFlag=training_type, training_flag=training_flag)
    update_parser(opts=opts.data_set_obj, attr='weight_modulation_factor', new_value=weight_modulation_factor)
    # if model_type is not model_type.MLP and model_type is not Model_type.ResNet32:
    #    Change_opts(opts=opts)
    model = create_model(opts=opts, model_type=model_type)
    if training_type is training_type.Full_ResNet and load_pretrained_model:
        Load_pretrained_model(opts, model, ds_type=ds_type, model_type=model_type)
    if load:
        project_path = str(Path(__file__).parents[1])
        models_path = os.path.join(project_path, 'data/models')
        model_path = 'Cifar10_additive_model_larger/ResNet_epoch50.pt'
        path = os.path.join(models_path, model_path)
        checkpoint = torch.load(path)[
            'model_state_dict']  # Loading the saved data.
        model.load_state_dict(checkpoint)

    train_dl, val_dl, test_dl = Get_dataloaders(opts=opts, task_id=task_id)
    name = Get_Model_path(opts, ds_type, training_type, training_flag, model_type, task_id)
    learned_params = Get_Learned_Params(model, training_flag, task_id)
    wrapped_model = ModelWrapped(opts=opts, model=model, learned_params=learned_params,
                                 task_id=task_id)
    load_model(model=wrapped_model, results_dir=os.path.join(opts.results_dir,
                                                             'CIFAR10/Train_original_resnet/Full_Model_Training/ResNet20/SGD'),
               model_path='thre_0.0_0_factor_0.1_wd_0.0001_bs_128_lr=_0.1_milestones_[100, '
                          '150]_drop_out_rate_0.2_modulation_factor_[4, 4, 1, 1]_channels_[16, 16, 32, '
                          '64]modulation_False/Model_best.ckpt')
    # print(wrapped_model.Accuracy(test_dl))

    trainer = Define_Trainer(opts, name)
    trainer.fit(wrapped_model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    trainer.test(wrapped_model, test_dl)
    # print(trainer.test(wrapped_model, test_dl,verbose = False)[0]['test_acc_epoch'])


for i in range(1, 6):
    for factor in [[2, 4, 1, 1], [4, 4, 1, 1], [2, 2, 1, 1], [4, 8, 1, 1], [8, 4, 1, 1], [8, 8, 1, 1]]:
        train_classification(task_id=i, training_flag=TrainingFlag.Modulation, load=False,
                             ds_type=data_set_types.CIFAR100,
                             training_type=ModelFlag.Full_ResNet, model_type=Model_type.ResNet20,
                             load_pretrained_model=False,
                             weight_modulation_factor=factor)
