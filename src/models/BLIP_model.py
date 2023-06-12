from typing import Any
import pytorch_lightning as pl
from transformers import BlipConfig, BlipForImageTextRetrieval


class BLIPWrapper(pl.LightningModule):
    def __init__(self, config, model_url_or_path="microsoft/blip-1000", *args: Any,
                 **kwargs: Any) -> None:
        """
        __init__ _summary_

        Args:
            config (_type_): _description_
            model_url_or_path (str, optional): _description_. Defaults to "microsoft/blip-1000".
        """
        super().__init__(*args, **kwargs)
        self.model = BlipForImageTextRetrieval.from_pretrained(pretrained_model_name_or_path=model_url_or_path)
        self.model_config = config
        self.huggingface_config = BlipConfig.from_pretrained(model_url_or_path)
