import logging

from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import Resize

from common.common_classes import MeasurementsBase, DataInputs
from common.common_functions import get_loss_function
from transformers.configs.config import Config
from transformers.vit_models_only_forward.AllModelsWithOnlyForward import AllModelsWithOnlyForward
from transformers.vit_models_only_forward.vitModelInterface import VitModelInterface


class ObjectTaskInterface(object):
    """Interface for object tasks."""

    task_name: str = None
    task_dataset: str = None
    measurements = None
    dataset_function = None
    output_size: int = 1000
    made_model_transformer = None
    transforms_resize: Resize = None
    number_of_instructions: int = 0

    def __init__(self, weights_losses=None):
        self.model_only_forward: VitModelInterface = None
        self.loss_function = get_loss_function(weights_losses)

    def reshape_input(self, model_input_shape) -> Resize:
        """Reshape input."""
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        """Forward function."""
        raise NotImplementedError

    def set_loss_function(self, weight_losses):
        self.loss_function = get_loss_function(weight_losses)

    def set_weights_losses(self, data_loader: DataLoader, inputs_to_struct, number_classes: int, logger: logging.Logger,
                           number_of_batches: int, config: Config):
        pass

    def validate(self, task_name: str, task_dataset: str) -> bool:
        """Validate task name and dataset."""
        return self.task_name == task_name and self.task_dataset == task_dataset

    def loss(self, inputs: DataInputs, outs: Tensor, number_classes: int, is_pretrain: bool = None) -> Tensor:
        """Loss function."""
        raise NotImplementedError

    def basic_loss_activation(self, taskk_out, label_taskk, weights_losses=None):
        loss_function = self.loss_function
        if weights_losses:
            loss_function = get_loss_function(weights_losses)
        return loss_function(taskk_out, label_taskk)

    def accuracy(self, outs, inputs: DataInputs, nclasses, is_pretrain: bool = None):
        """Accuracy function."""
        raise NotImplementedError

    def add_name(self, measurement: MeasurementsBase):
        """Add name to measurements."""
        raise NotImplementedError

    def another_layers(self, *args):
        """Another layers."""
        return None

    def put_model(self, model_implementation: str, model_name: str):  #
        self.model_only_forward: VitModelInterface = AllModelsWithOnlyForward.get_wanted_model(model_name)

        if self.model_only_forward is not None:
            self.model_only_forward.edit_model_implementation(model_implementation)
        else:
            raise ValueError("model_implementation not found")

    def get_number_of_instructions(self):
        return self.number_of_instructions
