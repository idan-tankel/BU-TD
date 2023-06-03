"""
Train classification data-sets continually.
"""

from src.data.Parser import GetParser
from src.python_lightning.Wrapper import ModelWrapped
from src.Modules.Models_Loading import Load_Pretrained_model, load_model
from src.Utils import num_params, Get_dataloaders, \
    Get_Model_path, Get_Learned_Params, Define_Trainer
from src.data.Enums import data_set_types, Model_type, TrainingFlag
from src.Modules.Create_model import create_model

from src.python_lightning.losses import Get_loss_fun


def train_classification(ds_type: data_set_types,
                         training_flag: TrainingFlag,
                         model_type=Model_type.ResNet18, load_pretrained_model: bool = True,
                         ) -> None:
    """
    Train cifar continual learning.
    Args:
        ds_type: The data-set layer_type.
        training_flag: The training flag.
        model_type: The model layer_type,
        load_pretrained_model: Load pretrained model.

    """
    task_id = ds_type.id()
    opts = GetParser(ds_type=ds_type, model_type=model_type, training_flag=training_flag)
    model = create_model(opts=opts, model_type=model_type)
    if load_pretrained_model:
        Load_Pretrained_model(opts, model)
    learned_params = Get_Learned_Params(model, training_flag, task_id)
    loss_fun = Get_loss_fun(training_flag=training_flag, opts=opts, model=model,
                            learned_params=learned_params)

    train_dl, test_dl = Get_dataloaders(opts=opts, task_id=task_id)
    old_name, name = Get_Model_path(opts, ds_type, training_flag, model_type, task_id)

    wrapped_model = ModelWrapped(opts=opts, model=model, learned_params=learned_params,
                                 task_id=task_id, name=name, loss_fun=loss_fun, training_flag=training_flag)
    trainer = Define_Trainer(opts, name)

    path = '/home/sverkip/data/BU-TD/transfer_learning/data/models/StanfordCars/LWF/MobileNetV3/Adam/[6, ' \
           '6]/Task_1_threshold_0.005_bs_32_lr=_0.0001_milestones_[15, 30]_modulation_factor_[6, 6]_interpolation_bicubic_lambda_5000_wd_0.0'
    state_dict = load_model(results_dir=path, model_path='val_acc=0.780.ckpt')

    wrapped_model.load_state_dict(state_dict)

    trainer.fit(wrapped_model, train_dataloaders=train_dl, val_dataloaders=test_dl)
    # trainer.test(wrapped_model, test_dl)

    opts.ds_type = 'ImageNet'
    train_dl, test_dl = Get_dataloaders(opts=opts, task_id=0)
    trainer.test(wrapped_model, test_dl)


train_classification(training_flag=TrainingFlag.LWF, ds_type=data_set_types.Sketches,
                     model_type=Model_type.MobileNetV3)
