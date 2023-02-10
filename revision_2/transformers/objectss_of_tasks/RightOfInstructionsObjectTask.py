import torch
from torch import Tensor
from torchvision.transforms import Resize

from common.common_classes import DataInputs
from transformers.Utils.RightOfUtils import get_labels_task_right_of
from transformers.objectss_of_tasks.RightOfObjectTask import RightOfObjectTask


class RightOfInstructionsObjectTask(RightOfObjectTask):
    task_name: str = 'right_of_instructions'
    number_of_instructions = 1

    def __init__(self, weights_losses=None, number_classes: int = 47, model_input_shape: int = 224):
        super().__init__(weights_losses)
        self.resize_instructions_layer: torch.nn.Linear = None
        self.model_input_shape = None
        self.resize_layer = None
        self.number_classes = number_classes
        self.output_size = number_classes + 1

    def reshape_input(self, model_input_shape: tuple) -> Resize:
        self.model_input_shape = model_input_shape

        return super().reshape_input(model_input_shape)

    def another_layers(self, num_instructions: int, model):
        # im_shape = self.model_input_shape[0]
        self.resize_instructions_layer = torch.nn.Linear(self.number_classes, num_instructions)
        return torch.nn.ModuleDict({"resize_instructions_layer": self.resize_instructions_layer})

    def forward(self, inputs: DataInputs, task_added_layers: torch.nn.ModuleDict):
        # Reshape the instructions
        instructions = self.instructions_handle(inputs)

        x = self.model_only_forward.forward(self.transforms_resize(inputs.images), self.made_model_transformer,
                                            instructions)
        return x

    # Get the instructions - and move them through the added layers
    def instructions_handle(self, inputs: DataInputs):
        instructions = self.get_instructions(inputs)
        return self.resize_instructions_layer(instructions).unsqueeze(1)

    def get_instructions(self, inputs: DataInputs):
        instructions = inputs.flag[:, -self.number_classes:]
        return instructions

    def loss(self, inputs: DataInputs, outs, number_classes, is_pretrain: bool = None) -> Tensor:
        """
        losses function - calculate loss - for each image - then mean over images


        :param outs: output of the model - shape (batch_size, num_classes)
        :param inputs: the inputs - DataInputs
        :param number_classes: number of classes
        :param is_pretrain: if it is pretrain
        """

        wanted_output = self.get_wanted_output(inputs, is_pretrain=is_pretrain)
        loss_task = self.loss_function(outs, wanted_output)

        return loss_task.mean()

    def accuracy(self, outs, inputs: DataInputs, nclasses, is_pretrain: bool = None) -> [float, float]:
        # TODO - fill the right accuracy
        wanted_output: Tensor = self.get_wanted_output(inputs, is_pretrain=is_pretrain)
        outputs: Tensor = outs.argmax(1)
        return (wanted_output == outputs).float().mean()

    def get_wanted_output(self, inputs: DataInputs, is_pretrain: bool = None) -> Tensor:
        if is_pretrain:
            return inputs.label_all.squeeze(1)[:, 0]
        else:
            labels_task = get_labels_task_right_of(inputs.label_all.squeeze(1), self.number_classes)
            wanted_output = labels_task[self.get_instructions(inputs) == 1]
        return wanted_output
