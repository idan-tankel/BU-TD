import os.path
from enum import Enum, auto
from pathlib import Path

from supp.loss_and_accuracy import multi_label_loss_weighted, multi_label_loss, multi_label_accuracy, \
    multi_label_accuracy_weighted
from supp.utils import get_omniglot_dictionary
from typing import Union, Callable


class Flag(Enum):
    """
    The possible training flags.
    """
    TD = auto()  # ordinary BU-TD network training.
    NOFLAG = auto()  # NO TASK network, should output all adjacent neighbors.
    CL = auto()  # Continual learning flag, a BU-TD network with allocating task embedding for each task as in the paper.


class DsType(Enum):
    """
    The Data Set types.
    """
    Emnist = auto()  # Emnist dataset.
    FashionMnist = auto()  # Fashion-mnist dataset.
    Omniglot = auto()  # Omniglot dataset.
    Cifar10 = auto()  # Cifar10.
    Cifar100 = auto()  # Cifar100.

    def Enum_to_name(self):
        """
        Returns: Returning for each DataSet type its name.
        """
        if self is DsType.Emnist:
            return "emnist"
        if self is DsType.FashionMnist:
            return "fashionmnist"
        if self is DsType.Omniglot:
            return "omniglot"


class GenericDataset:
    def __init__(self, flag_at: Flag, ds_type: DsType, ndirections: int = 4):
        """
        Generic Specification convenient for all datasets.

        Args:
            flag_at: The model flag.
            ds_type: The data-set type flag.
        """
        self.flag = flag_at
        self.bu2_loss: Callable = multi_label_loss_weighted if flag_at is Flag.NOFLAG else multi_label_loss  # The BU2 classification loss according to the flag.
        self.task_accuracy: Callable = multi_label_accuracy_weighted if flag_at is Flag.NOFLAG else multi_label_accuracy  # The task accuracy metric according to the flag.
        self.project_path: Path = Path(__file__).parents[2]  # The path to the project.
        self.train_arg_emb: bool = False  # Whether for each task train also the argument(character) embedding, Only in Omniglot True.
        self.initial_tasks: list = [
            0]  # The initial tasks we first train in the zero forgetting continual learning experiments.
        self.ntasks: int = 1  # The number of tasks, in mnist,fashionmnist 1 for Omniglot 51.
        self.nclasses: Union[list] = [47 for _ in range(
            ndirections)]  # The number of classes for each task, 47 for mnist, 10 for fashion and for Omniglot its dictionary.
        self.results_dir = os.path.join(self.project_path, 'data/{}/results'.format(
            ds_type.Enum_to_name()))  # The trained model directory.
        self.use_bu1_loss = True if flag_at is not Flag.NOFLAG else False  # Whether to use the bu1 loss only for Omniglot always False.
        self.ndirections = ndirections  # The number of directions we query about, default 4


class EmnistDataset(GenericDataset):
    def __init__(self, flag_at: Flag, ndirections: int = 4):
        """
        Emnist Dataset.
        Args:
            flag_at: The model flag.
            ndirections: The number of directions.
        """
        super(EmnistDataset, self).__init__(flag_at=flag_at, ds_type=DsType.Emnist, ndirections=ndirections)


class FashionmnistDataset(GenericDataset):
    def __init__(self, flag_at: Flag, ndirections: int = 4):
        """
        Fashionmnist dataset.
        The same as emnist but with 10 classes.
        Args:
            flag_at: The model flag.
            ndirections: The number of directions.
        """
        super(FashionmnistDataset, self).__init__(flag_at, ds_type=DsType.FashionMnist, ndirections=ndirections)
        self.nclasses = 10  # The same as emnist, just we have just 10 classes in the dataset.


class OmniglotDataset(GenericDataset):
    def __init__(self, initial_tasks, flag_at: Flag, ndirections: int = 4):
        """
        Omniglot dataset.
        Args:
            flag_at: The model flag.
            ndirections: The number of directions.
        """
        super(OmniglotDataset, self).__init__(flag_at, ds_type=DsType.Omniglot, ndirections=ndirections)
        self.initial_tasks = initial_tasks  # The initial tasks set.
        self.ntasks = 51  # We have 51 tasks.
        raw_data_path = os.path.join(self.project_path, 'data/omniglot/RAW')  # The raw data path.
        if not os.path.exists(raw_data_path):
            raise "You didn't download the raw omniglot data."
        self.nclasses = get_omniglot_dictionary(self.initial_tasks,
                                                raw_data_path)  # Computing for each language its number of charatcres.
        self.use_bu1_loss = False  # As there are many classes we don't use the bu1 loss.
        self.train_arg_emb = True  # As we have 50 different languages, we need for each language its argument embedding.


class AllOptions:
    def __init__(self, ds_type: DsType, flag_at: Flag, ndirections: int = 4, initial_task_for_omniglot: list = []):
        """
        Given dataset type returns its associate Dataset specification.
        Args:
            ds_type: The dataset type.
            flag_at: The training flag.
            ndirections: The number of directions.
            initial_task_for_omniglot: The initial task list for Omniglot only.
        """
        if ds_type is DsType.Emnist:
            self.data_obj = EmnistDataset(flag_at=flag_at, ndirections=ndirections)  # Emnist.

        if ds_type is DsType.FashionMnist:
            self.data_obj = FashionmnistDataset(flag_at=flag_at, ndirections=ndirections)  # Fashionmnist

        if ds_type is DsType.Omniglot:
            self.data_obj = OmniglotDataset(flag_at=flag_at, ndirections=ndirections,
                                            initial_tasks=initial_task_for_omniglot)  # Omniglot.
