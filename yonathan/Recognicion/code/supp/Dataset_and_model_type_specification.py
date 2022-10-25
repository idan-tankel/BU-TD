import os.path
from enum import Enum, auto
from pathlib import Path

from supp.loss_and_accuracy import multi_label_loss_weighted, multi_label_loss, multi_label_accuracy, \
    multi_label_accuracy_weighted


class Flag(Enum):
    """
    The possible training flags.

    """
    TD = auto()  # ordinary BU-TD network.
    NOFLAG = auto()  # NO TASK network.
    ZF = auto()  # BU-TD network with allocating task embedding for each task.


class DsType(Enum):
    """
    The Data Set types.
    """
    Emnist = auto()  # Emnist.
    FashionMnist = auto()  # Fashion mnist.
    Omniglot = auto()  # Omniglot.
    Cifar10 = auto()  # Cifar10.
    Cifar100 = auto()  # Cifar100.

    def Enum_to_name(self):
        if self is DsType.Emnist:
            return "emnist"
        if self is DsType.FashionMnist:
            return "fashionmnist"
        if self is DsType.Omniglot:
            return "omniglot"


class inputs_to_struct:
    # class To make a structured input out of list of tensors
    # TODO add to this classes METHODS that modifies that input
    def __init__(self, inputs):
        """
        class To make a structured input out of list of tensors
        Since the data is kept as a dictionary, in the future this class will be DEPRECATED
        Args:
            inputs(`[torch.Tensor]`): The tensor list of the following scheme
            img, label_task, flag, label_all, label_existence    
        """
        img, label_task, flag, label_all, label_existence = inputs
        self.image = img
        self.label_all = label_all
        self.label_existence = label_existence
        self.label_task = label_task
        self.flag = flag

    def struct2list(self):
        """
        struct2list replaces the old sturct_to_input function as a METHOD of the class.
        Returns: The list of the input tensors of the following scheme
            img, label_task, flag, label_all, label_existence

        Returns:
            _type_: _description_
        """
        return [self.image, self.label_task, self.flag, self.label_all, self.label_existence]

    def compute_flag(self):
        # TODO add a flag to the class that is computed from the label_task
        pass


def folder_size(path: str) -> int:
    """
    Returns the number of files in a given folder.
    Args:
        path: Path to a language file.

    Returns: Number of files in the folder
    """
    return len([_ for _ in os.scandir(path)])


def create_dict(path: str) -> dict:
    """
    Args:
        path: Path to all raw Omniglot languages.

    Returns: Dictionary of number of characters per language

    """
    dict_language = {}
    cnt = 0
    # for language in Omniglot_raw find the number of characters in it.
    for ele in os.scandir(path):
        # Find number of characters in the folder.
        dict_language[cnt] = folder_size(ele)
        cnt += 1
    return dict_language


def get_omniglot_dictionary(initial_tasks: list, ntasks: int, raw_data_folderpath: str) -> list:
    """
    Getting the omniglot dictionary.
    Args:
        initial_tasks: The initial tasks set.
        ntasks: The number of tasks.
        raw_data_folderpath: The path to the raw data.

    Returns: A dictionary assigning for each task its number of characters.

    """
    nclasses_dictionary = {}
    dictionary = create_dict(raw_data_folderpath)
    nclasses_dictionary[0] = sum(
        dictionary[task] for task in initial_tasks)  # receiving number of characters in the initial tasks.
    nclasses = []
    for i in range(ntasks - 1):  # copying the number of characters for all classes
        nclasses_dictionary[i + 1] = dictionary[i]
    for i in range(ntasks):  # creating nclasses according to the dictionary and num_chars
        nclasses.append(nclasses_dictionary[i])
    return nclasses


class GenericDataset:
    def __init__(self, flag_at: Flag):
        """
        Args:
            flag_at: The model flag.
        """
        self.flag = flag_at
        # The BU2 classification loss.
        self.bu2_loss = multi_label_loss_weighted if flag_at is Flag.NOFLAG else multi_label_loss
        # The task accuracy metric.
        self.task_accuracy = multi_label_accuracy_weighted if flag_at is Flag.NOFLAG else multi_label_accuracy
        self.project_path = Path(__file__).parents[2]
        # Whether for each task train also the argument(character) embedding.
        self.train_arg_emb = False
        # The initial tasks we first train in the zero forgetting experiments.
        self.initial_tasks = [0]
        self.ntasks = None  # The number of tasks.
        self.nclasses = None  # The number of classes for each task.
        self.results_dir = None  # The trained model directory.
        self.use_bu1_loss = None  # Whether to use the bu1 loss.
        self.ndirections = None  # The number of directions we query about.


class EmnistDataset(GenericDataset):
    def __init__(self, flag_at, ndirections=4):
        super(EmnistDataset, self).__init__(flag_at)
        self.ntasks = 4  # As we have one language(emnist) we have one task.
        # We have 47 different characters.
        self.nclasses = [47 for _ in range(ndirections)]
        self.results_dir = os.path.join(
            self.project_path, 'data/emnist/results')
        self.use_bu1_loss = True if flag_at is not Flag.NOFLAG else False
        self.ndirections = ndirections


class FashionmnistDataset(EmnistDataset):
    def __init__(self, flag_at, ndirections=4):
        super(FashionmnistDataset, self).__init__(
            flag_at, ndirections=ndirections)
        # The same as emnist, just we have just 10 classes in the dataset.
        self.nclasses = [10 for _ in range(ndirections)]


class OmniglotDataset(GenericDataset):
    def __init__(self, flag_at, ndirections):
        super(OmniglotDataset, self).__init__(flag_at)
        self.initial_tasks = [27, 5, 42, 18, 33]  # The initial tasks set.
        self.ntasks = 51  # We have 51 tasks.
        self.results_dir = os.path.join(
            self.project_path, 'data/omniglot/results')
        raw_data_path = os.path.join(self.project_path, 'data/omniglot/RAW')
        if not os.path.exists(raw_data_path):
            raise "You didn't download the raw omniglot data."
        # Computing for each langauge its numbe of charatcres.
        self.nclasses = get_omniglot_dictionary(
            self.initial_tasks, self.ntasks, raw_data_path)
        # As there are many classes we don't use the bu1 loss.
        self.use_bu1_loss = False
        self.ndirections = ndirections
        self.train_arg_emb = True


class AllOptions(GenericDataset):
    def __init__(self, ds_type, flag_at, ndirections=4):
        if ds_type is DsType.Emnist:
            self.data_obj = EmnistDataset(flag_at, ndirections=ndirections)

        if ds_type is DsType.FashionMnist:
            self.data_obj = FashionmnistDataset(
                flag_at, ndirections=ndirections)

        if ds_type is DsType.Omniglot:
            self.data_obj = OmniglotDataset(flag_at, ndirections=ndirections)
