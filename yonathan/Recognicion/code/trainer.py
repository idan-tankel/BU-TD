import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from supp.general_functions import create_optimizer_and_sched
import torch
from supp.Parser import GetParser
from supp.get_dataset import get_dataset_for_spatial_realtions
from supp.FlagAt import Flag, DsType, Model_Options_By_Flag_And_DsType
from pytorch_lightning import loggers



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
        self.log("loss",loss)
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
            samples =opts.inputs_to_struct(batch)
            _ , task_accuracy = self.accuracy(outs, samples)
            return task_accuracy

    def configure_optimizers(self):
        opti, sched = create_optimizer_and_sched(self.opts, self.learned_params)
        return {"optimizer":opti, "lr_scheduler":sched}

    def validation_epoch_end(self, outputs):
        return torch.cat(outputs,dim=0).sum() / len(outputs)

    def test_epoch_end(self, outputs):
        return torch.cat(outputs, dim=0).sum() / len(outputs)





opts = Model_Options_By_Flag_And_DsType(Flag=Flag.ZF, DsType=DsType.Emnist)
parser = GetParser(opts=opts, language_idx=0,direction = 'left')
data_path = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/samples/6_extended_testing_new_changes_beta_0'
# Create the data for right.
[the_datasets, train_dl ,  test_dl, val_dl ,] = get_dataset_for_spatial_realtions(parser, data_path,lang_idx = 0, direction = 0)

learned_params = parser.model.module.parameters()
tb_logger = loggers.TensorBoardLogger(save_dir = "/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/results/MyFirstlogger")
trainer = pl.Trainer(accelerator='gpu',max_epochs =30,logger=tb_logger, default_root_dir="/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/results/MyFirstCheckpint.ckpt")
model = MyModel(parser,learned_params)
trainer.fit(model, train_dataloaders = train_dl, val_dataloaders = test_dl)
# Goals:
# 1. having train -> test - > val.
# 2. have full logger.
# 3. save model in checkpoints by test.
# 4. Get rid of training_functions, measurmnets.