"""
Here we define for each data-set its parameters like number of classes,
number of direction, tasks.
"""
import os.path
from enum import Enum, auto
from pathlib import Path
from typing import Callable
from typing import Union

import numpy as np

from Data_Creation.Create_dataset_classes import DsType  # Import the Data_Creation set types.
from training.Metrics.Accuracy import multi_label_accuracy, multi_label_accuracy_weighted
from training.Metrics.Loss import multi_label_loss_weighted, multi_label_loss
from training.Utils import get_omniglot_dictionary, tuple_direction_to_index
import argparse


# Define the Flag Enums, and Dataset specification and baseline methods.


class Flag(Enum):
    """
    The possible training flags.
    """
    TD = auto()  # Ordinary BU-TD network training.
    NOFLAG = auto()  # Non-guided model, should output for each character its adjacent neighbor.
    CL = auto()  # Continual learning flag, a BU-TD network with allocating task embedding for each task.


class RegType(Enum):
    """
    The possible baselines.
    """
    EWC = 'EWC'
    LWF = 'LWF'
    LFL = 'LFL'
    MAS = 'MAS'
    RWALK = 'RWALK'
    SI = 'SI'
    IMM_Mode = 'IMM_Mode'
    IMM_Mean = 'IMM_Mean'
    Naive = 'Naive'

    def __str__(self):
        return self.value

    def class_to_reg_factor(self, opts: argparse) -> float:
        """
        Given the model options, get the associated regularization factor.
        Args:
            opts: The model opts.

        Returns: The regularization factor

        """
        return getattr(opts, self.__str__() + '_lambda')


class GenericDataParams:
    """
    Generic Dataset parameters Specification convenient for all datasets.
    """

    def __init__(self, flag_at: Flag, ds_type: DsType, num_x_axis: int = 1, num_y_axis: int = 1):
        """
        Args:
            flag_at: The model flag.
            ds_type: The data-set type flag.
            num_x_axis: The neighbor levels in the x-axis.
            num_y_axis: The neighbor levels in the x-axis.
        """
        self.flag: Flag = flag_at
        self.bu2_loss: Callable = multi_label_loss_weighted if flag_at is Flag.NOFLAG \
            else multi_label_loss  # The BU2 classification loss according to the flag.
        self.task_accuracy: Callable = multi_label_accuracy_weighted if flag_at is Flag.NOFLAG \
            else multi_label_accuracy  # The task Accuracy metric according to the flag.
        self.project_path: Path = Path(__file__).parents[3]  # The path to the project.
        self.initial_directions: list = [
            (1, 0)]  # The initial tasks we first train in our experiments, default to right direction
        self.ntasks: int = 1  # The number of tasks, in mnist, Fashionmnist it's 1, for Omniglot 51.
        self.num_x_axis: int = num_x_axis  # Number of directions we want to generalize to in the x-axis.
        self.num_y_axis: int = num_y_axis  # Number of directions we want to generalize to in the y-axis.
        self.ndirections: int = (2 * self.num_x_axis + 1) * (
                2 * self.num_y_axis + 1)  # The number of directions we query about.
        # The number of classes for each task, 47 for mnist, 10 for fashion and for Omniglot its dictionary.
        # self.ndirections = 1
        self.nclasses: dict = {i: 47 for i in range(self.ndirections)}
        self.results_dir: str = os.path.join(self.project_path,
                                             f'data/{str(ds_type)}/results/')  # The trained model directory.
        self.baselines_dir: str = os.path.join(self.project_path,
                                               f'data/{str(ds_type)}/Baselines/')
        self.Data_specific_path = os.path.join(self.project_path, 'data/{}'.format(str(ds_type)))
        self.use_bu1_loss: bool = True if flag_at is not Flag.NOFLAG \
            else False  # Whether to use the bu1 loss.
        self.num_heads: list = [1 for _ in range(self.ndirections)]
        self.image_size: np.array = None  # The image size, will be defined later.


class EmnistDataset(GenericDataParams):
    """
    The Emnist dataset specification.
    """

    def __init__(self, flag_at: Flag):
        """
        Args:
            flag_at: The model flag.

        """
        super(EmnistDataset, self).__init__(flag_at=flag_at, ds_type=DsType.Emnist, num_x_axis=2, num_y_axis=2)
        self.image_size = [130, 200]  # The Emnist image size.
        # The initial indexes.
        self.initial_directions = [(1, 0)]
        initial_directions = [
            tuple_direction_to_index(num_x_axis=self.num_x_axis, num_y_axis=self.num_y_axis, direction=direction,
                                     ndirections=self.ndirections)[0]
            for direction in self.initial_directions]
        # The number of heads.
        self.num_heads = [1 if direction not in initial_directions else len(self.initial_directions) for direction in
                          range(self.ndirections)]


class FashionmnistDataset(GenericDataParams):
    """
    The Fashionmnist dataset specification.
    """

    def __init__(self, flag_at: Flag):
        """
        Fashionmnist dataset.
        The same as Emnist but with 10 classes.
        Args:
            flag_at: The model flag.
        """
        super(FashionmnistDataset, self).__init__(flag_at=flag_at, ds_type=DsType.Fashionmnist)
        self.nclasses = {i: 10 for i in
                         range(self.ndirections)}  # The same as Emnist, just we have just 10 classes in the dataset.
        self.image_size = [112, 130]  # The FashionMnist image size.


class OmniglotDataset(GenericDataParams):
    """
    The Omniglot specification.
    """

    def __init__(self, initial_tasks: int, flag_at: Flag):
        """
        Omniglot dataset.
        Args:
            initial_tasks: The initial task desired set.
            flag_at: The model flag.

        """
        super(OmniglotDataset, self).__init__(flag_at=flag_at, ds_type=DsType.Omniglot,
                                              num_y_axis=0)  # No Up/Down relations, just Right/Left.
        self.initial_tasks = initial_tasks  # The number of languages we want to use.
        self.ntasks = 51  # We have 51 tasks.
        self.use_bu1_loss = False  # As there are many classes we don't use the bu1 loss.
        self.image_size = [55, 200]  # The Omniglot image size.
        raw_data_path = os.path.join(self.project_path, 'data/Omniglot/RAW/omniglot-py/Unified')  # The raw data path.
        self.nclasses = get_omniglot_dictionary(self.initial_tasks,
                                                raw_data_path)  # Computing for each language its number of characters.
        self.ndirections = 5
        self.num_heads = [1 for _ in range(self.ndirections)]


class AllDataSetOptions:
    """
    Class saving for the desired data-set its data object.
    """

    def __init__(self, ds_type: DsType, flag_at: Flag,
                 initial_task_for_omniglot_only: Union[int, None] = None):
        """
        Given dataset type returns its associate Dataset specification.
        Args:
            ds_type: The dataset type.
            flag_at: The training flag.
            initial_task_for_omniglot_only: The initial task list only for Omniglot.
        """
        if ds_type is DsType.Emnist:
            self.data_obj = EmnistDataset(flag_at=flag_at)  # Emnist.

        if ds_type is DsType.Fashionmnist:
            self.data_obj = FashionmnistDataset(flag_at=flag_at)  # Fashionmnist

        if ds_type is DsType.Omniglot:
            self.data_obj = OmniglotDataset(flag_at=flag_at,
                                            initial_tasks=initial_task_for_omniglot_only)  # Omniglot.
