"""
Here we define used structs including input to struct, out to struct, and training flag.
"""
import argparse

import torch.nn as nn
from torch import Tensor
from ..Utils import tuple_direction_to_index


class inputs_to_struct:
    """
    Class receiving list of input tensors and makes to a class.
    """

    def __init__(self, inputs: list[Tensor]):
        """
        Args:
            inputs: The tensor list.
        """
        img, label_task, flag, label_all, label_existence, index, *rest = inputs
        self.image = img  # The image.
        self.label_all = label_all  # The label all.
        self.label_existence = label_existence  # The label existence.
        self.label_task = label_task.squeeze()  # The label task.
        self.flags = flag  # The flag.
        self.index = index


class outs_to_struct:
    """
    Struct transforming the model output list to struct.
    """

    def __init__(self, outs: list[Tensor]):
        """
        Struct transforming the model output list to struct.
        Args:
            outs: The model outs.
        """
        occurrence_out, bu_features, bu2_features, classifier = outs
        self.occurrence_out = occurrence_out  # The occurrence output.
        self.classifier = classifier  # The classes output.
        self.bu = bu_features  # The BU1 output
        self.features = bu2_features  # The features output.


class Training_flag:
    """
    For continual learning, we freeze some layers.
    For that, we have created a class containing which layer groups we want to train.
    """

    def __init__(self, opts: argparse, train_all_model: bool = False, train_arg: bool = False,
                 train_task_embedding: bool = False,
                 train_head: bool = False):
        """
        Args:
            opts: The model options.
            train_all_model: Whether to train all model.
            train_arg: Whether to train arg.
            train_task_embedding: Whether to train the task embedding.
            train_head: Whether to train the read-out head.
        """
        self.train_all_model = train_all_model
        self.train_arg = train_arg
        self.task_embedding = train_task_embedding
        self.head_learning = train_head
        self.opts = opts

    def Get_learned_params(self, model: nn.Module, task_idx: int, direction: tuple[int, int]):
        """
        Given model, task id, direction id we return the desired trainable parameters.
        Args:
            model: The model.
            task_idx: Language index.
            direction: The direction id.

        Returns: The desired parameters.
        """

        direction_idx, idx = tuple_direction_to_index(num_x_axis=self.opts.num_x_axis, num_y_axis=self.opts.num_y_axis,
                                                      direction=direction,
                                                      ndirections=self.opts.ndirections, task_id=task_idx)
        learned_params = []
        if self.train_all_model:
            # Train all model.
            learned_params = list(model.parameters())
            return learned_params

        if self.task_embedding:
            # Training the task embedding associate with the task.
            learned_params.extend(model.TE[direction_idx])
        if self.head_learning:
            # Train the task-head associated with the task, direction.
            learned_params.extend(model.transfer_learning[task_idx][direction_idx])
        if self.train_arg:
            # Train the argument embedding associated with the task.
            learned_params.extend(model.tdmodel.argument_embedding[task_idx])

        return learned_params


class Task_to_struct:
    """
    Struct to contain the task, direction.
    """

    def __init__(self, task: int, direction: tuple):
        """
        Args:
            task: The task.
            direction: The direction.
        """
        self.task = task
        self.direction = direction
        self.unified_task = (task, direction)
