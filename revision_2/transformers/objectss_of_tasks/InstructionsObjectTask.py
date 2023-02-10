import torch
from torch.nn import ModuleList

from common.common_classes import DataInputs
from transformers.objectss_of_tasks.RightOfInstructionsObjectTask import RightOfInstructionsObjectTask


class InstructionsObjectTask(RightOfInstructionsObjectTask):
    task_name: str = 'instructions'
    number_of_instructions = 2
    size_of_instruction_digit_inner_embedding_layer = 10  # TODO - is from config
    size_of_instruction_direction_inner_embedding_layer = 10

    len_of_instruction_direction_in_token = 64
    len_of_instruction_digit_in_token = 64
    instructions_digit_embedding_layers: torch.nn.ModuleList = None
    instructions_direction_embedding_layers: torch.nn.ModuleList = None

    def __init__(self, weights_losses=None, number_classes: int = 47, model_input_shape: int = 224):
        super().__init__(weights_losses, number_classes, model_input_shape)

    def get_wanted_output(self, inputs: DataInputs, is_pretrain: bool = None):
        if is_pretrain:
            return super().get_wanted_output(inputs, True)
        instructions_digit, instructions_direction = self.get_instructions(inputs)

        # instructions_direction for each one - if it is 1 in location 0 it should be -1, and if in location 1 its 1
        # it should be 1
        instructions_direction = instructions_direction.argmax(1) * 2 - 1
        instructions_digit = instructions_digit.argmax(1)

        # find location of instructions_digit in inputs.label_existence
        location_of_digit_in_picture = inputs.label_all.squeeze(1) == instructions_digit.unsqueeze(1)
        location_of_task = location_of_digit_in_picture.float().argmax(1) + instructions_direction

        # Now find the wanted outputs that in location location_of_task(but if smaller than 0 or larger than number of
        # digits in the image - should be (number_classes + 1))
        not_in_image = torch.where((location_of_task < 0) | (location_of_task >= 6), False,
                                   False)
        location_of_task = torch.where((location_of_task < 0) | (location_of_task >= 6), 0,
                                       location_of_task)

        # Get location location_of_task in inputs.label_all
        wanted_output = inputs.label_all.squeeze(1).gather(1, location_of_task.unsqueeze(1))

        # Remove out of the image
        wanted_output[not_in_image] = self.number_classes
        return wanted_output.squeeze(1)

    def another_layers(self, num_instructions: int, model):
        self.len_of_instruction_direction_in_token = model.pos_embed.shape[2]  # * model.pos_embed.shape[3]
        self.len_of_instruction_digit_in_token = model.pos_embed.shape[2]  # * model.pos_embed.shape[3]
        self.create_instruction_digit_embedding()
        self.create_instruction_direction_embedding()
        return torch.nn.ModuleDict({"instructions_digit": self.instructions_digit_embedding_layers,
                                    "instructions_direction": self.instructions_direction_embedding_layers})

    def create_instruction_direction_embedding(self):
        self.instructions_direction_embedding_layers = \
            torch.nn.ModuleList([torch.nn.Linear(2, self.size_of_instruction_direction_inner_embedding_layer),
                                 torch.nn.Linear(self.size_of_instruction_direction_inner_embedding_layer,
                                                 self.len_of_instruction_direction_in_token)])

    def create_instruction_digit_embedding(self):
        self.instructions_digit_embedding_layers = \
            torch.nn.ModuleList(  # TODO - maybe to remove one layer - to be only one layer
                [torch.nn.Linear(self.number_classes, self.size_of_instruction_digit_inner_embedding_layer),
                 torch.nn.Linear(self.size_of_instruction_digit_inner_embedding_layer,
                                 self.len_of_instruction_digit_in_token)])

    def embed_instructions(self, instructions_digit):
        return self.move_through_module_list(instructions_digit, self.instructions_digit_embedding_layers)

    @staticmethod
    def move_through_module_list(inputs, module_list: ModuleList):
        x = inputs
        for layer in module_list:
            x = layer(x)
        return x

    def embed_direction(self, instructions_direction):
        return self.move_through_module_list(instructions_direction, self.instructions_direction_embedding_layers)

    # Get the instructions - and move them through the added layers
    def instructions_handle(self, inputs: DataInputs):
        instructions_digit, instructions_direction = self.get_instructions(inputs)
        return self.embed_instructions(instructions_digit), self.embed_direction(instructions_direction)

    def get_instructions(self, inputs):
        instructions_digit = inputs.flag[:, -self.number_classes:]
        instructions_direction = inputs.flag[:, :-self.number_classes]
        return instructions_digit, instructions_direction
