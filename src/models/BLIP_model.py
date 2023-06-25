from typing import Any, Optional
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from transformers import BlipConfig, BlipForImageTextRetrieval, AutoProcessor
import torch
from torch import device, cuda
from torch.optim import Adam
import torch.nn.functional as F


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
        image, text_embeddings, y = batch["img"], batch["text_embedding"], batch["label_all"]
        # preprocessing the example
        # shold be done in the dataset
        # image.to(device("cuda") if cuda.is_available() else device("cpu"))
        # the devices are handled via pytorch lightning accelerator arguemnt
        # the shape of the text_embedding should be `(batch_size, number_of_verbs_in_the_example, word_preprocessing_max_length)`
        # if you wish `(bs,c,word_preprocessing_max_length)` where `c` is the number of classes
        text_embeddings = torch.unbind(text_embeddings, dim=1)
        predictions = [
            self.model(**{"pixel_values": image, "input_ids": text_embedding})
            for text_embedding in text_embeddings
        ]
        itm_scores = [predictions.itm_score for predictions in predictions]
        itm_scores = torch.stack(itm_scores, dim=1)
        # shape of the hidden states is (batch_size,number_of_verbs,2)
        # where the predictions is whenever an example is positive for that label or not
        # see also https://huggingface.co/transformers/model_doc/blip.html#blipforimagetextretrieval
        # use the itm_head for the prediction
        # construct a binary cross entropy loss
        # according to original ALBEF code here https://github.com/salesforce/ALBEF/blob/b9727e43c3040491774d1b22cc27718aa7772fac/models/model_retrieval.py#LL161C10-L161C10
        #         itm_output = model(image,caption,match_head='itm')
        # itm_score = torch.nn.functional.softmax(itm_output,dim=1)[:,1]
        # from the tutorial https://github.com/salesforce/BLIP/blob/main/demo.ipynb
        itm_positive_score = F.softmax(itm_scores, dim=2)[:, :, 1]
        # now do regular CE loss with the matrix, but that it's not necessarily what they ment to do
        accuracy = torch.nn.functional.cross_entropy(itm_positive_score, torch.ones_like(itm_positive_score))
        loss = torch.nn.CrossEntropyLoss()(itm_scores.permute(0, 2, 1), torch.ones_like(itm_positive_score).to(torch.long))
        #  since the shape is (bs,number_of_verbs_in_the_example,2)
        # we assume that each example is positive for it's label
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('acc', accuracy, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, *args, **kwargs) -> STEP_OUTPUT | None:
        return super().validation_step(*args, **kwargs)

    def test_step(self, *args, **kwargs) -> STEP_OUTPUT | None:
        return super().test_step(*args, **kwargs)

    def configure_optimizers(self):
        batch_size = 10
        optimizer = Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer = optimizer,
            base_lr = 0.0001,
            max_lr = 0.002,
            step_size_up = batch_size//2,
            step_size_down = None,
            mode = 'triangular',
            gamma = 1.0,
            scale_fn = None,
            scale_mode = 'cycle',
            cycle_momentum = False,
            last_epoch = -1)
        monitor='val_loss'
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": monitor}
