import os.path
from enum import Enum, auto
from pathlib import Path
from typing import Callable
from typing import Union

from training.Metrics.Accuracy import multi_label_accuracy, multi_label_accuracy_weighted
from training.Metrics.Loss import multi_label_loss_weighted, multi_label_loss
from training.Utils import get_omniglot_dictionary, tuple_direction_to_index


class Flag(Enum):
    """
    The possible training flags.
    """
    TD = auto()  # ordinary BU-TD network training.
    NOFLAG = auto()  # NO TASK network, should output all adjacent neighbors(pure ResNet).
    CL = auto()  # Continual learning flag, a BU-TD network with allocating task embedding for each task.


class DsType(Enum):
    """
    The Data Set types.
    """
    Emnist = auto()  # Emnist dataset.
    FashionMnist = auto()  # Fashion-mnist dataset.
    Omniglot = auto()  # Omniglot dataset.

    def Enum_to_name(self):
        """
     Returns: Returning for each DataSet type its name.
     """

        if self is DsType.Emnist:
            return "Emnist"
        if self is DsType.FashionMnist:
            return "Fashionmnist"
        if self is DsType.Omniglot:
            return "Omniglot"


class GenericDataParams:
    def __init__(self, flag_at: Flag, ds_type: DsType, num_x_axis: int = 1, num_y_axis: int = 1):
        """
        Generic Dataset parameters Specification convenient for all datasets.

        Args:
            flag_at: The model flag.
            ds_type: The data-set type flag.
            num_x_axis: The neighbor levels in the x-axis.
            num_y_axis: The neighbor levels in the x-axis.
        """
        self.flag = flag_at
        self.bu2_loss: Callable = multi_label_loss_weighted if flag_at is Flag.NOFLAG \
            else multi_label_loss  # The BU2 classification loss according to the flag.
        self.task_accuracy: Callable = multi_label_accuracy_weighted if flag_at is Flag.NOFLAG \
            else multi_label_accuracy  # The task accuracy metric according to the flag.
        self.project_path: Path = Path(__file__).parents[3]  # The path to the project.
        self.initial_tasks: list = []  # The initial tasks we first train in our experiments.
        self.ntasks: int = 1  # The number of tasks, in mnist, Fashionmnist it's 1, for Omniglot 51.

        self.num_x_axis = num_x_axis  # Number of directions we want to generalize to in the x-axis.
        self.num_y_axis = num_y_axis  # Number of directions we want to generalize to in the y-axis.
        # TODO - CHANGE TO
        #  self.ndirections = (2 * self.num_x_axis + 1) * (
        #          2 * self.num_y_axis + 1) # The number of directions we query about.
        self.ndirections = 15
        # The number of classes for each task, 47 for mnist, 10 for fashion and for Omniglot its dictionary.
        self.nclasses: dict = {i: 47 for i in range(self.ndirections)}
        self.results_dir = os.path.join(self.project_path, 'data/{}/results'.format(
            ds_type.Enum_to_name()))  # The trained model directory.
        self.use_bu1_loss = True if flag_at is not Flag.NOFLAG \
            else False  # Whether to use the bu1 loss only for Omniglot always False.

        self.num_heads = [1 for _ in range(self.ndirections)]
        self.image_size = None  # The image size, will be defined later.


class EmnistDataset(GenericDataParams):
    def __init__(self, flag_at: Flag):
        """
        Emnist Dataset.
        Args:
            flag_at: The model flag.

        """
        super(EmnistDataset, self).__init__(flag_at=flag_at, ds_type=DsType.Emnist, num_x_axis=2)
        self.image_size = [130, 160]  # The Emnist image size.
        self.initial_tasks = [(1, 0), (-1, 0)]  # The initial tasks in Emnist(Right & Left).
        self.ndirections = self.ndirections
        # The initial indexes.
        initial_directions = [
            tuple_direction_to_index(self.num_x_axis, self.num_y_axis, direction, ndirections=self.ndirections,
                                     task_id=0)[0]
            for direction in self.initial_tasks]
        # The number of heads.
        self.num_heads = [1 if direction not in initial_directions else len(self.initial_tasks) for direction in
                          range(self.ndirections)]


class FashionmnistDataset(GenericDataParams):
    def __init__(self, flag_at: Flag):
        """
        Fashionmnist dataset.
        The same as Emnist but with 10 classes.
        Args:
            flag_at: The model flag.
        """
        super(FashionmnistDataset, self).__init__(flag_at, ds_type=DsType.FashionMnist)
        self.nclasses = {i: 10 for i in
                         range(self.ndirections)}  # The same as Emnist, just we have just 10 classes in the dataset.
        self.image_size = [112, 130]  # The FashionMnist image size.
        self.initial_tasks: list = [(1, 0)]  # The initial tasks(Right).


class OmniglotDataset(GenericDataParams):
    def __init__(self, initial_tasks: list, flag_at: Flag):
        """
        Omniglot dataset.
        Args:
            flag_at: The model flag.

        """
        super(OmniglotDataset, self).__init__(flag_at, ds_type=DsType.Omniglot,
                                              num_y_axis=0)  # No Up/Down relations, just Right/Left.
        self.initial_tasks = [initial_tasks]  # The initial tasks set.
        self.ntasks = 51  # We have 51 tasks.
        raw_data_path = os.path.join(self.project_path, 'data/omniglot/RAW')  # The raw data path.
        if not os.path.exists(raw_data_path):
            raise "You didn't download the raw omniglot data."
        self.nclasses = get_omniglot_dictionary(self.initial_tasks,
                                                raw_data_path)  # Computing for each language its number of characters.
        self.use_bu1_loss = False  # As there are many classes we don't use the bu1 loss.
        self.image_size = [55, 200]  # The Omniglot image size.


class AllOptions:
    def __init__(self, ds_type: DsType, flag_at: Flag,
                 initial_task_for_omniglot_only: Union[list, None] = None):
        """
        Given dataset type returns its associate Dataset specification.
        Args:
            ds_type: The dataset type.
            flag_at: The training flag.
            initial_task_for_omniglot_only: The initial task list for Omniglot only.
        """
        if ds_type is DsType.Emnist:
            self.data_obj = EmnistDataset(flag_at=flag_at)  # Emnist.

        if ds_type is DsType.FashionMnist:
            self.data_obj = FashionmnistDataset(flag_at=flag_at)  # Fashionmnist

        if ds_type is DsType.Omniglot:
            self.data_obj = OmniglotDataset(flag_at=flag_at,
                                            initial_tasks=initial_task_for_omniglot_only)  # Omniglot.
