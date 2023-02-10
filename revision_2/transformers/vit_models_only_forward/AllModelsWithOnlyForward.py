from typing import List

from transformers.vit_models_only_forward.pit import Pit
from transformers.vit_models_only_forward.vitModelInterface import VitModelInterface


class AllModelsWithOnlyForward:
    all_models_only_forward: List[VitModelInterface] = []

    @classmethod
    def create_all_object_tasks(cls):
        if not cls.all_models_only_forward:
            cls.all_models_only_forward.append(Pit())

    @classmethod
    def get_wanted_model(cls, model_parent_name: str) -> VitModelInterface:
        AllModelsWithOnlyForward.create_all_object_tasks()
        for model in cls.all_models_only_forward:
            if model.validate(model_parent_name):
                return model
        return None
