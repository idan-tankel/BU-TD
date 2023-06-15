from typing import Any, Optional
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from transformers import BlipConfig, BlipForImageTextRetrieval, AutoProcessor
import torch
from torch import device, cuda
from torch.optim import Adam


class BLIPWrapper(pl.LightningModule):
    def __init__(self, config, model_url_or_path="microsoft/blip-1000", *args: Any,
                 **kwargs: Any) -> None:
        """
        __init__ _summary_

        Args:
            config (BU_TD.config): The model configuration of the local project.
            model_url_or_path (str, optional): The model configuration from huggingface. 
            Defaults to "microsoft/blip-1000".
        """
        super().__init__()  # without any arguments in order not to pass something unexpected
        if config.Models.pretrained_model_name is not None:
            model_url_or_path = config.Models.pretrained_model_name
        model = BlipForImageTextRetrieval.from_pretrained(pretrained_model_name_or_path=model_url_or_path)
        model.to(device("cuda") if cuda.is_available() else device("cpu"))
        self.model = model
        self.model_config = config
        self.huggingface_config = BlipConfig.from_pretrained(model_url_or_path)
        self.preprocessor = AutoProcessor.from_pretrained(pretrained_model_name_or_path=model_url_or_path)
        

    def training_step(self, batch, batch_index, *args, **kwargs) -> STEP_OUTPUT:
        image, text, y = batch["img"], batch["text"], batch["label_all"]
        # preprocessing the example
        # shold be done in the dataset
        # image.to(device("cuda") if cuda.is_available() else device("cpu"))
        # the devices are handled via pytorch lightning accelerator arguemnt
        inputs = {"pixel_values": image, "input_ids": y.to(torch.int64)}
        predictions = self.model(**inputs)
        return super().training_step(*args, **kwargs)

    def validation_step(self, *args, **kwargs) -> STEP_OUTPUT | None:
        return super().validation_step(*args, **kwargs)

    def test_step(self, *args, **kwargs) -> STEP_OUTPUT | None:
        return super().test_step(*args, **kwargs)

    def configure_optimizers(self):
        batch_size = 10
        optimizer = Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer=optimizer,
            base_lr=0.0001,
            max_lr=0.002,
            step_size_up=batch_size//2,
            step_size_down=None,
            mode='triangular',
            gamma=1.0,
            scale_fn=None,
            scale_mode='cycle',
            cycle_momentum=False,
            last_epoch=-1)
        monitor = 'val_loss'
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": monitor}
