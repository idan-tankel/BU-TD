"""
Train classification data-sets continually.
"""
import torch.nn

from src.data.Parser import get_parser
from src.Modules.Models.Wrapper import ModelWrapped
from src.Utils import num_params, get_dataloaders, \
    get_model_path, get_learned_params, define_trainer,  load_pretrained_model
from src.data.Enums import DataSetTypes, TrainingFlag
from src.data.losses import LwFLoss, OrdinaryLoss
from src.Modules.Models.MobileNet import MobileNetV2
from src.Modules.Continual_blocks.Batch_norm import store_running_stats
def train_classification(ds_type: DataSetTypes,
                         training_flag: TrainingFlag,
                         load_pretrained_model: bool = True,
                         ) -> None:
    """
    Train cifar continual learning.
    Args:
        task_id: The task id in [0,10]
        ds_type: The data-set layer_type.
        training_flag: The training flag.
        model_type: The model layer_type,
        load_pretrained_model: Load pretrained model.

    """
    task_id = ds_type.id()
    opts = get_parser(ds_type=ds_type, training_flag=training_flag)
    model = MobileNetV2(opts=opts)
    learned_params = get_learned_params(model, training_flag, task_id)

    train_dl, test_dl = get_dataloaders(opts=opts, task_id=task_id)
    model_path = get_model_path(opts, ds_type, training_flag)

    wrapped_model = ModelWrapped(opts=opts, model=model, learned_params=learned_params,
                                 task_id=task_id, name=model_path)

    trainer = define_trainer(opts, model_path)
    trainer.fit(wrapped_model, train_dataloaders=train_dl, val_dataloaders=test_dl)
    trainer.test(wrapped_model, test_dl)

train_classification(training_flag=TrainingFlag.Classifier_Only, ds_type=DataSetTypes.StanfordCars,
                     load_pretrained_model=True)
