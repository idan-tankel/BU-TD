"""
Train classification data-sets continually.
"""
from src.data.Parser import GetParser
from src.python_lightning.Wrapper import ModelWrapped
from src.Utils import num_params, Load_Pretrained_model, Get_dataloaders, \
    Get_Model_path, Get_Learned_Params, Define_Trainer, load_model
from src.data.Enums import data_set_types, Model_type, TrainingFlag
from src.Modules.Create_model import create_model


def train_classification(task_id: int, ds_type: data_set_types,
                         training_flag: TrainingFlag,
                         model_type=Model_type.ResNet18, load_pretrained_model: bool = True,
                         ) -> None:
    """
    Train cifar continual learning.
    Args:
        task_id: The task id in [0,10]
        ds_type: The data-set type.
        training_flag: The training flag.
        training_type: The model type.
        model_type: The model type,
        load_pretrained_model: Load pretrained model.

    """
    opts = GetParser(ds_type=ds_type, model_type=model_type, training_flag=training_flag)
    model = create_model(opts=opts, model_type=model_type)
    if load_pretrained_model:
        Load_Pretrained_model(opts, model, ds_type=ds_type, model_type=model_type)

    train_dl, val_dl, test_dl = Get_dataloaders(opts=opts, task_id=task_id)
    old_name, name = Get_Model_path(opts, ds_type, training_flag, model_type, task_id)
    learned_params = Get_Learned_Params(model, training_flag, task_id)
    wrapped_model = ModelWrapped(opts=opts, model=model, learned_params=learned_params,
                                 task_id=task_id, name=name)

    trainer = Define_Trainer(opts, name)
    trainer.fit(wrapped_model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    trainer.test(wrapped_model, test_dl)


train_classification(task_id=5, training_flag=TrainingFlag.Masks, ds_type=data_set_types.Food101,
                     model_type=Model_type.MobileNet)
