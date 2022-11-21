import argparse

import torch
import torch.nn as nn

from training.Utils import tuple_direction_to_index


# Here we define all used structs including input to struct, out to struct, and training flag.

class inputs_to_struct:
    # class receiving list of tensors and makes to a class.
    def __init__(self, inputs: tuple[torch]):
        """
        Args:
            inputs: The tensor list.
        """
        img, label_task, flag, label_all, label_existence = inputs
        self.image = img
        self.label_all = label_all
        self.label_existence = label_existence
        self.label_task = label_task
        self.flag = flag

class outs_to_struct:
    def __init__(self, outs: list[torch]):
        """
        Struct transforming the model output list to struct.
        Args:
            outs: The model outs.
        """
        occurrence_out, bu_features, bu2_features, classifier = outs
        self.occurrence_out = occurrence_out
        self.classifier = classifier
        self.bu = bu_features
        self.features = bu2_features


class Training_flag:
    def __init__(self, parser: argparse, train_all_model: bool = False, train_arg: bool = False,
                 train_task_embedding: bool = False,
                 train_head: bool = False):
        """
        Args:
            train_all_model: Whether to train all model.
            train_arg: Whether to train arg.
            train_task_embedding: Whether to train the task embedding.
            train_head: Whether to train the read-out head.
        """
        self.train_all_model = train_all_model
        self.train_arg = train_arg
        self.task_embedding = train_task_embedding
        self.head_learning = train_head
        self.parser = parser

    def Get_learned_params(self, model: nn.Module, task_idx: int, direction: tuple[int, int]):
        """
        Args:
            model: The model.
            task_idx: Language index.
            direction: The direction.

        Returns: The desired parameters.
        """

        direction_idx, idx = tuple_direction_to_index(self.parser.num_x_axis, self.parser.num_y_axis, direction,
                                                      self.parser.ndirections, task_idx)
        learned_params = []
        if self.task_embedding:
            # Training the task embedding associate with the direction.
            learned_params.extend(model.TE[direction_idx])
        if self.head_learning:
            # Train the task-head associated with the task, direction.
            learned_params.extend(model.transfer_learning[idx])
        if self.train_arg:
            # Train the argument embedding associated with the task.
            learned_params.extend(model.tdmodel.argument_embedding[task_idx])
        if self.train_all_model:
            learned_params = list(model.parameters())
        return learned_params
