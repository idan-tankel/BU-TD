import os
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from training.Data.Checkpoints import CheckpointSaver
from training.Data.Data_params import Flag, DsType
from training.Data.Get_dataset import get_dataset_for_spatial_relations
from training.Data.Model_Wrapper import ModelWrapped
from training.Data.Parser import GetParser
from training.Data.Structs import Training_flag
from training.Modules.Create_Models import create_model
from training.Modules.Models import BUTDModel
from training.Utils import num_params


def main(train_right, train_left, ds_type=DsType.Emnist, flag=Flag.CL, model_type=BUTDModel, lang_id=0,
         direction=(1, 0)):
    parser = GetParser(model_flag=flag, ds_type=ds_type, model_type=model_type)
    project_path = Path(__file__).parents[1]
    results_dir = os.path.join(project_path, 'data/{}/results/model'.format(str(ds_type)))
    data_set_path = os.path.join(project_path, f"data/{str(ds_type)}")
    data_path = os.path.join(data_set_path, f'samples/(1,6)_data_set_matrix{str(lang_id)}')
    Checkpoint_saver = CheckpointSaver(dirpath=results_dir + f"Model_lang={lang_id}_direction={direction}_small_emb")
    wandb_path = os.path.join(data_set_path, 'logging/wandb')
    wandb_logger = WandbLogger(project="My_first_project_5.10", job_type='train', save_dir=wandb_path)
    trainer = pl.Trainer(accelerator='gpu', max_epochs=parser.EPOCHS, logger=wandb_logger)
    model = create_model(parser)

    if train_right:
        training_flag = Training_flag(parser, train_all_model=True)
        direction = (1, 0)
        # learned_params = list(model.parameters())
        learned_params = training_flag.Get_learned_params(model, task_idx=lang_id, direction=direction)
        DataLoaders = get_dataset_for_spatial_relations(parser, data_path, lang_idx=lang_id, direction_tuple=direction)
        wrapped_model = ModelWrapped(parser, model, learned_params, check_point=Checkpoint_saver,
                                     direction_tuple=direction,
                                     task_id=lang_id,
                                     nbatches_train=len(DataLoaders['train_dl']), train_ds=DataLoaders['train_ds'])
        wrapped_model.load_model(model_path='Right_long/BUTDModel_epoch71_direction=(1, 0).pt')
        # print(wrapped_model.Accuracy(DataLoaders['test_dl']))
        trainer.fit(wrapped_model, train_dataloaders=DataLoaders['train_dl'], val_dataloaders=DataLoaders['test_dl'])

    if train_left:
        training_flag = Training_flag(parser, train_task_embedding=True)
        learned_params = training_flag.Get_learned_params(model, task_idx=lang_id, direction=direction)
        #  print(learned_params)
        DataLoaders = get_dataset_for_spatial_relations(parser, data_path, lang_idx=lang_id,
                                                        direction_tuple=[direction])
        wrapped_model = ModelWrapped(parser, model, learned_params, check_point=Checkpoint_saver,
                                     direction_tuple=[direction],
                                     task_id=lang_id,
                                     nbatches_train=len(DataLoaders['train_dl']), train_ds=DataLoaders['train_ds'])

        #        wrapped_model.load_model(model_path='modelModel_lang=49_direction=(-1, 0)/BUTDModel_best_direction=(-1, 0).pt')
        print(num_params(learned_params))
        acc = wrapped_model.Accuracy(DataLoaders['test_dl'])
        print(acc)
        #   print(Accuracies, sum(Accuracies),len(Accuracies))
        trainer.fit(wrapped_model, train_dataloaders=DataLoaders['train_dl'], val_dataloaders=DataLoaders['test_dl'])


main(False, True, ds_type=DsType.Omniglot, lang_id=49, direction=(-1, 0))
