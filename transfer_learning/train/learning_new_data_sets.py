"""
Train classification data-sets continually.
"""
from src.data.Parser import GetParser
from src.python_lightning.Wrapper import ModelWrapped
from pathlib import Path
import os
import torch
from src.Utils import num_params, Load_pretrained_model, Change_opts, Get_dataloaders, \
    Get_Model_path, Get_Learned_Params, Define_Trainer, load_model
from src.data.Enums import data_set_types, Model_type, TrainingFlag, ModelFlag
from src.Modules.Create_model import create_model


def train_classification(task_id: int, load: bool, ds_type: data_set_types,
                         training_flag: TrainingFlag, ModelFlag: ModelFlag,
                         model_type=Model_type.ResNet18, load_pretrained_model: bool = True) \
        -> None:
    """
    Train cifar continual learning.
    Args:
        task_id: The task id in [0,10]
        load: Whether to load pretrained model.
        ds_type: The data-set type.
        training_flag: The training flag.
        ModelFlag: The model type.
        model_type: The model type,
        load_pretrained_model: Load pretrained model.

    """
    opts = GetParser(ds_type=ds_type, model_type=model_type, ModelFlag=ModelFlag)
    if model_type is not model_type.MLP and model_type is not Model_type.ResNet32:
        Change_opts(opts=opts)
    model = create_model(opts=opts, model_type=model_type)
    if ModelFlag is ModelFlag.Full_ResNet and load_pretrained_model:
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
    name = Get_Model_path(opts, ds_type, ModelFlag, training_flag, model_type, task_id)
    learned_params = Get_Learned_Params(model, training_flag, task_id)
    wrapped_model = ModelWrapped(opts=opts, model=model, learned_params=learned_params,
                                 task_id=task_id)
    '''
    load_model(model=wrapped_model, results_dir='/home/sverkip/data/tran/data/models/CIFAR100/Train_original_resnet'
                                                f'/Modulation_Training/ResNet32/Adam/New_'
                                                f'{i}_factor_0.2_wd_0.0_bs_96_lr=_0.1_milestones_[50, 100]_drop_out_rate_0.2_modulation_factor_[4, 4, 1, 1]_channels_[16, 16, 32, 64]modulation_True'
                                                ,
               model_path='Model_best.ckpt')
    '''
    # print(wrapped_model.Accuracy(test_dl))

    trainer = Define_Trainer(opts, name)
    trainer.fit(wrapped_model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    # trainer.test(wrapped_model, test_dl)
    print(trainer.test(wrapped_model, test_dl,verbose = False)[0]['test_acc_epoch'])

for i in range(1,11):
    train_classification(task_id=i, training_flag=TrainingFlag.Modulation, load=False, ds_type=data_set_types.CIFAR100,
                     ModelFlag=ModelFlag.Full_ResNet, model_type=Model_type.ResNet32, load_pretrained_model=False)

