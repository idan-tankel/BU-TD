import argparse
import os

import pytorch_lightning as pl
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from supp.Dataset_and_model_type_specification import Flag, DsType, Model_Options_By_Flag_And_DsType
from supp.Parser import GetParser
from supp.general_functions import create_optimizer_and_sched
from supp.get_dataset import get_dataset_for_spatial_realtions
from supp.logger import print_detail
from supp.measurments import Measurements
from supp.measurments import set_datasets_measurements
from supp.training_functions import fit

# NO SEED in data_functions and not in blocks.
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class MyModel(LightningModule):
    def __init__(self, opts, learned_params):
        super().__init__()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        self.model = opts.model
        self.opts = opts
        self.loss_fun = opts.loss_fun
        self.learned_params = learned_params
        self.accuracy = opts.task_accuracy

    def training_step(self, batch, batch_idx):
        outs = self.model(batch)
        loss = self.loss_fun(self.model,batch, outs)
        self.log("loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            outs = self.model(batch)
            loss = self.loss_fun(batch, outs)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            outs = self.model(batch)
            outs = self.model.module.outs_to_struct(outs)
            samples = self.opts.inputs_to_struct(batch)
            _ , task_accuracy = self.accuracy(outs, samples)
            return task_accuracy

    def configure_optimizers(self):
        opti, sched = create_optimizer_and_sched(self.opts, self.learned_params)
        return {"optimizer":opti, "lr_scheduler":sched}

    def validation_epoch_end(self, outputs):
        print(torch.cat(outputs,dim=0).sum() / len(outputs))
        return torch.cat(outputs,dim=0).sum() / len(outputs)

    def test_epoch_end(self, outputs):
        return torch.cat(outputs, dim=0).sum() / len(outputs)

class Training_flag:
    def __init__(self, train_all_model: bool, train_arg: bool, task_embedding: bool, head_learning: bool):
        """
        Args:
            train_all_model: Whether to train all model.
            train_arg: Whether to train arg.
            task_embedding: Whether to train the task embedding.
            head_learning: Whether to train the read out head.
        """
        self.train_all_model = train_all_model
        self.train_arg = train_arg
        self.task_embedding = task_embedding
        self.head_learning = head_learning

    def Get_learned_params(self, model: nn.Module, lang_idx: int, direction: int):
        """
        Args:
            model: The model.
            lang_idx: Language index.
            direction: The direction.

        Returns: The desired parameters.

        """
        if self.train_all_model:
            return list(model.parameters())
        learned_params = []
        idx = direction + 4 * lang_idx
        if self.task_embedding:
            learned_params.extend(model.module.task_embedding[direction])
        if self.head_learning:
            learned_params.extend(model.module.transfer_learning[idx])
        if self.train_arg:
            learned_params.extend(model.module.argument_embedding[lang_idx])
        return learned_params

def train_omniglot(opts: argparse, lang_idx: int, the_datasets: list, training_flag: Training_flag, direction: int):
    """
    Args:
        opts: The model options.
        lang_idx: The language index.
        the_datasets: The datasets.
        training_flag: The training flag.
        direction: The direction.

    Returns: The optimum and the learning history.

    """
    set_datasets_measurements(the_datasets, Measurements, opts, opts.model)
    cudnn.benchmark = True
    # Deciding which parameters will be trained: if True all the model otherwise,only the task embedding.
    learned_params = training_flag.Get_learned_params(opts.model, lang_idx, direction)
    opts.optimizer, opts.scheduler = create_optimizer_and_sched(opts, learned_params)
    # Training the learned params of the model.
    return fit(opts, the_datasets, lang_idx, direction)


def name(index):
    if index == -1:
        return "5R"
    else:
        return "6_extended_" + str(index)


def main_omniglot(lang_idx: int = -1, train_right: bool = True, train_left: bool = True):
    """
    Args:
        lang_idx:
        train_right: Whether to train right.
        train_left: Whether to train left.

    Returns: None.
    """
    opts = Model_Options_By_Flag_And_DsType(Flag=Flag.ZF, DsType=DsType.Omniglot)
    parser = GetParser(opts=opts, language_idx=lang_idx)
    print_detail(parser)
    data_path = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/omniglot/samples_new/' + name(lang_idx)
    # Create the data for right.
    [the_datasets, _, test_dl, _,_,_,_ ] = get_dataset_for_spatial_realtions(parser, data_path, lang_idx=lang_idx + 1,
                                                                        direction=0)
    # Training Right.
    path_loading = os.path.join('Model5R', 'model_right_best.pt')
    model_path = parser.results_dir
   # load_model(parser.model, model_path, path_loading, load_optimizer_and_schedular=False)
    # load_running_stats(parser.model, task_emb_id = 1);
    #  acc = accuracy(parser.model, test_dl)
    #   print("Done training right, with accuracy : " + str(acc))
    if train_right:
        parser.EPOCHS = 60
        training_flag = Training_flag(train_all_model=True, train_arg=True, task_embedding=False, head_learning=True)
        train_omniglot(parser, lang_idx=lang_idx + 1, the_datasets=the_datasets, training_flag=training_flag,  direction=0)

    if train_left:
        parser.EPOCHS = 100
        [the_datasets, _, _, _] = get_dataset_for_spatial_realtions(parser, data_path, lang_idx=lang_idx + 1,
                                                                    direction=1)
        training_flag = Training_flag(train_all_model=False, train_arg=False, task_embedding=False, head_learning=True)
        train_omniglot(parser, lang_idx=lang_idx + 1, the_datasets=the_datasets, training_flag=training_flag,  direction=1)

def TrainLanguage(opts: argparse, lang_idx: int, the_dataloaders: list, training_flag: Training_flag, direction: int):
    """
    Args:
        opts: The model options.
        lang_idx: The language index.
        the_datasets: The datasets.
        training_flag: The training flag.
        direction: The direction.

    Returns: The optimum and the learning history.
    """
    # Create the data for right.
    tmpdir = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/results/MyFirstCheckPoint.ckpt'
    learned_params = training_flag.Get_learned_params(opts.model, lang_idx, direction)
    model_ckpt = ModelCheckpoint(dirpath=tmpdir, monitor=None, save_top_k=-1, save_last=True)
    tb_logger = loggers.TensorBoardLogger(save_dir = "/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/results/MyFirstlogger")
    trainer = pl.Trainer(accelerator='gpu',max_epochs = opts.epochs,logger=tb_logger, callbacks =[model_ckpt])
    model = MyModel(opts,learned_params)
    #
    train_dl = the_dataloaders['train']
    test_dl = the_dataloaders['test']
    val_dl = the_dataloaders['val']
    #
    trainer.fit(model, train_dataloaders = train_dl, val_dataloaders = val_dl, test_dataloaders = test_dl )


main_omniglot(35, True, True)
