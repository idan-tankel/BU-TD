from matplotlib import transforms
from torch import nn, device, cuda
from transformers import ViTConfig, ViTModel, TrainingArguments, ViTForImageClassification,ViTFeatureExtractor
from vit_pytorch import cct
from torchvision.transforms import transforms
from Configs.Config import Config
from types import SimpleNamespace
# local imports
try:
    from supp.Dataset_and_model_type_specification import inputs_to_struct
except ImportError:
    from ..supp.Dataset_and_model_type_specification import inputs_to_struct
try:
    from .Heads import MultiLabelHead,OccurrenceHead
except ImportError:
    from Heads import MultiLabelHead,OccurrenceHead
from typing import Union
import yaml


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

    def __init__(self, config):
        """
        __init__ _summary_

        Args:
            config (`Configs.Config.config`): The model configuration - gloabl configuration. In case that the config is None, some default values are used from empty ViTConfig object.
        """
        super().__init__()
        dev = device("cuda" if cuda.is_available() else "cpu")
        # self.model = ViTForImageClassification(config=config).to(dev)
        self.cct = cct.CCT(
            img_size=224,
            embedding_dim=config.Models.nfilters[-1],
            num_classes=config.Models.nfilters[-1],
            n_conv_layers=2,
            kernel_size=3,
            stride=2,
            padding=3,
            pooling_kernel_size=3,
            pooling_stride=2,
            pooling_padding=1,
            num_layers=4,
            num_heads=2,
            mlp_ratio=1.,
            
        )
        self.taskhead = MultiLabelHead(opts=config)
        self.model = nn.Sequential(self.cct, self.taskhead)
        self.model.to(dev)

    def document2attentionconfig(self, document, override=True) -> ViTConfig:
        """
        document2config method converts yaml document to `ViTConfig` object.
        The attributes of the yaml document should be the same as the `transformers.ViTConfig` object.
        For more information about the `transformers.ViTConfig` object, please visit https://huggingface.co/transformers/model_doc/vit.html#transformers.ViTConfig


        ## Attention! ##
        This function overrides the model under the `self.model` attribute.

        Args:
            document (dict): document

        Returns:
            ViTConfig: ViTConfig object
        """
        with open(document, 'r') as stream:
            config_as_dict = yaml.safe_load(stream)
        config = ViTConfig(**config_as_dict)
        if override:
            self.model = ViTForImageClassification(config=config)
        return config

    def forward(self, x):
        """
        forward method forward pass

        Args:
            x (torch.Tensor): input tensor - unstructured data

        Returns:
            torch.Tensor: output tensor
        """
        x = inputs_to_struct(x)
        model_inputs = x.image
        # TODO change this to support parameter and not hard coded
        model_inputs = self.cct(model_inputs)
        return self.taskhead(model_inputs)
    
    def outs_to_struct(self, outs):
        """
        outs_to_struct change the output of the forward pass to a structured output

        Args:
            outs (torch.Tensor): output of the forward pass

        Returns:
            SimpleNamespace: structured output
        """        
        task_out = outs
        outs_ns = SimpleNamespace(task=task_out)
        return outs_ns



class Attention2(ViTForImageClassification):
    """
    Attention2 Class is an implementation of BU-TD approach instead of BU approach
    """
    def __init__(self,cfg:Config):
        """
        __init__ _summary_

        Args:
            config (config): The Config class object - global model and trainig configuration
        """
        super().__init__(ViTConfig())
        self.feature_extractor =  ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        self.vit = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        # Now the self.vit is the ViT model and self.classifier is the built-in classifier        
        self.taskhead = MultiLabelHead(opts=cfg)
        self.vit.to(device("cuda" if cuda.is_available() else "cpu"))
        # save the model attribute
        self.model = nn.Sequential(self.vit, self.taskhead)



    def forward(self,x):
        """
        forward method forward pass

        Args:
            x (torch.Tensor): input tensor - unstructured data

        Returns:
            torch.Tensor: output tensor
        """
        x = inputs_to_struct(x)
        # the input to struct function is of the dataset - to crete a structured data
        model_inputs = x.image
        vit_outputs = self.vit(model_inputs).logits
        return self.taskhead(vit_outputs)
        # since the BaseModelOutputWithPooling is an output wrapper for the ViT model as documented here 
        # https://huggingface.co/transformers/v4.4.2/main_classes/output.html


    def outs_to_struct(self, outs):
        """
        outs_to_struct change the output of the forward pass to a structured output

        Args:
            outs (torch.Tensor): output of the forward pass

        Returns:
            SimpleNamespace: structured output
        """        
        task_out = outs
        outs_ns = SimpleNamespace(task=task_out)
        return outs_ns