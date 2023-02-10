import torch

from transformers.models.timm_models.pit import PoolingVisionTransformer
from transformers.vit_models_only_forward.vitModelInterface import VitModelInterface


class Pit(VitModelInterface):
    model_parent_name: str = 'pit'
    model_implementation: str = None

    # Modify the forward_features function - by adding the instructions
    def forward(self, inputs, pit_model: PoolingVisionTransformer, instructions):
        # Replacing the end of the cls_token with the instructions
        insert_instructions_in_cls_tokens = False
        cls_tokens = self.get_cls_tokens(inputs, insert_instructions_in_cls_tokens, instructions, pit_model)

        inputs = pit_model.patch_embed(inputs)
        inputs = pit_model.pos_drop(inputs + pit_model.pos_embed)

        inputs = self.add_the_instructions_as_tokens(inputs, instructions)

        inputs, cls_tokens = pit_model.transformers((inputs, cls_tokens))
        cls_tokens = pit_model.norm(cls_tokens)

        # Apply the forward_head
        cls_tokens = pit_model.forward_head(cls_tokens) # TODO - maybe here take only the first token for each image
        return cls_tokens

    @staticmethod
    def add_the_instructions_as_tokens(inputs, instructions):
        stacked_instructions = torch.stack(instructions, dim=2)
        stacked_instructions = stacked_instructions.permute(0, 2, 1)
        inputs = torch.cat([inputs, stacked_instructions.unsqueeze(3).repeat((1, 1, 1, inputs.shape[3]))], 1)
        return inputs

    @staticmethod
    def get_cls_tokens(inputs, insert_instructions_in_cls_tokens, instructions, pit_model):
        if insert_instructions_in_cls_tokens:
            cls_tokens = pit_model.cls_token[:, :, :-instructions.shape[2]].expand(inputs.shape[0], -1, -1)
            cls_tokens = torch.cat([cls_tokens, instructions], 2)
        else:
            cls_tokens = pit_model.cls_token.expand(inputs.shape[0], -1, -1)
        return cls_tokens

    # Edit the specific model name
    def edit_model_implementation(self, model_name: str):
        self.model_implementation = model_name
