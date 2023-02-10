from torch import nn


class VitModelInterface:
    model_parent_name: str = None
    model_implementation: str = None

    def validate(self, model_parent_name: str):
        return self.model_parent_name == model_parent_name

    def forward(self, inputs, edited_model: nn.Module, instructions):
        raise NotImplementedError

    def edit_model_implementation(self, model_name: str):
        self.model_implementation = model_name
