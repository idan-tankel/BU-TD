import time
from typing import Any

import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import TQDMProgressBar

from transformers.Utils.TrainingData import TrainData


class EpochProgressBar(TQDMProgressBar):
    basic_desc: str = None

    def __init__(self, train_data: TrainData, is_pretrain: bool = False):
        super().__init__()
        self.start_time = -1
        self.main_progress_bar = None
        self.train_data: TrainData = train_data
        self.is_pretrain = is_pretrain

    # Train #
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *_) -> None:
        super(EpochProgressBar, self).on_train_epoch_end(trainer, pl_module)
        self.refresh("Train")

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *_) -> None:
        super(EpochProgressBar, self).on_train_batch_end(trainer, pl_module)
        self.refresh("Train")

    def on_train_epoch_start(self, trainer: pl.Trainer, *_: Any):
        self.start_time = time.time()
        super(EpochProgressBar, self).on_train_epoch_start(trainer)
        self.basic_desc = self.main_progress_bar.desc
        self.refresh("Train")

    # validation #
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *_) -> None:
        super(EpochProgressBar, self).on_validation_epoch_end(trainer, pl_module)
        self.refresh("Validation")
        if self.start_time > 0:
            wandb.log({'Epoch time': (time.time() - self.start_time)})
            self.start_time = -1
        else:
            print("Error - start time is not set!!!")

    def on_validation_batch_end(self, trainer: pl.Trainer, *_) -> None:
        super(EpochProgressBar, self).on_validation_batch_end(trainer, *_)
        self.refresh("Validation")

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        print("")
        super(EpochProgressBar, self).on_validation_epoch_start(trainer, pl_module)
        self.main_progress_bar.set_description(f"Epoch {trainer.current_epoch}")
        self.basic_desc = self.main_progress_bar.desc
        self.refresh("Validation")

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *_) -> None:
        super(EpochProgressBar, self).on_test_epoch_end(trainer, pl_module)
        self.refresh("Test")

    def on_test_batch_end(self, trainer: pl.Trainer, *_) -> None:
        super(EpochProgressBar, self).on_test_batch_end(trainer, *_)
        self.refresh("Test")

    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        print("")
        super(EpochProgressBar, self).on_test_epoch_start(trainer, pl_module)
        self.main_progress_bar.set_description(f"Epoch {trainer.current_epoch}")
        self.basic_desc = self.main_progress_bar.desc
        self.refresh("Test")

    def refresh(self, dataset: str):
        if self.is_pretrain:
            self.main_progress_bar.set_postfix_str('pretrain, ' + str(self.train_data) + ",  " + time.asctime())
        else:
            self.main_progress_bar.set_postfix_str(str(self.train_data) + ",  " + time.asctime())
        self.main_progress_bar.set_description(dataset.ljust(10) + " " + self.basic_desc)
        # TODO - add config if want time for progress bar
