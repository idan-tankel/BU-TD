from matplotlib import transforms
from torch import nn, device, cuda
from transformers import ViTConfig, ViTModel, TrainingArguments, ViTForImageClassification
from typing import Union


class Attention(nn.Module):
    """
    Attention class represents attention block

    Attributes:
        config (ViTConfig): ViTConfig object
        num_heads (int): number of heads
        head_dim (int): head dimension
        hidden_dim (int): hidden dimension
        scale (float): scale
        qkv (torch.nn.Linear): linear layer
        attn_drop (torch.nn.Dropout): dropout layer
        proj (torch.nn.Linear): linear layer
        proj_drop (torch.nn.Dropout): dropout layer
    """

    def __init__(self, config: Union[ViTConfig, None]):
        """
        __init__ _summary_

        Args:
            config (`ViTConfig` | `None`): The model configuration in case of a not pretrained model. In case that the config is None, some default values are used from empty ViTConfig object.
        """
        super().__init__()
        dev = device("cuda" if cuda.is_available() else "cpu")
        self.model_pretrained = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224-in21k', num_labels=2).to(dev)
        if config == None:
            config = ViTConfig()
        self.model = ViTForImageClassification(config=config).to(dev)

    def document2config(self, document):
        """
        document2config method converts yaml document to `ViTConfig` object.
        The attributes of the yaml document should be the same as the `transformers.ViTConfig` object.
        For more information about the `transformers.ViTConfig` object, please visit https://huggingface.co/transformers/model_doc/vit.html#transformers.ViTConfig

        Args:
            document (dict): document

        Returns:
            ViTConfig: ViTConfig object
        """
        config = ViTConfig(**document)
        return config

    def forward(self, x):
        """
        forward method forward pass

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        return self.model(x)
