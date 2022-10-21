import os.path
from enum import Enum, auto
<<<<<<< HEAD
from pathlib import Path

=======

from supp.general_functions import get_omniglot_dictionary
>>>>>>> 315b11ac3016dc72662fd8ca96881ae68c5cda6d
from supp.loss_and_accuracy import multi_label_loss_weighted, multi_label_loss, multi_label_accuracy, \
    multi_label_accuracy_weighted


class Flag(Enum):
    """
<<<<<<< HEAD
    The possible training flags.

    """
    TD = auto() # ordinary BU-TD network.
    NOFLAG = auto() # NO TASK network.
    ZF = auto() # BU-TD network with allocating task embedding for each task.
=======
    The possible flags.
    """
    TD = auto()
    NOFLAG = auto()
    ZF = auto()
>>>>>>> 315b11ac3016dc72662fd8ca96881ae68c5cda6d

class DsType(Enum):
    """
    The Data Set types.
    """
<<<<<<< HEAD
    Emnist = auto() # Emnist.
    FashionMnist = auto() # Fashion mnist.
    Omniglot = auto() # Omniglot.
    Cifar10 = auto() # Cifar10.
    Cifar100 = auto() # Cifar100.

    def Enum_to_name(self):
        if self is DsType.Emnist:
            return "emnist"
        if self is DsType.FashionMnist:
            return "fashionmnist"
        if self is DsType.Omniglot:
            return "omniglot"

class inputs_to_struct:
    # class receiving list of tensors and makes to a class.
    def __init__(self, inputs):
        """
        Args:
            inputs: The tensor list.
        """
=======
    Emnist = auto()
    FashionMnist = auto()
    Omniglot = auto()
    Cifar10 = auto()
    Cifar100 = auto()

class inputs_to_struct:
    def __init__(self, inputs):
>>>>>>> 315b11ac3016dc72662fd8ca96881ae68c5cda6d
        img, label_task, flag, label_all, label_existence = inputs
        self.image = img
        self.label_all = label_all
        self.label_existence = label_existence
        self.label_task = label_task
        self.flag = flag

<<<<<<< HEAD
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
    for ele in os.scandir(path):  # for language in Omniglot_raw find the number of characters in it.
        dict_language[cnt] = folder_size(ele)  # Find number of characters in the folder.
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
    def __init__(self,flag_at:Flag):
        """
        Args:
            flag_at: The model flag.
        """
        self.flag = flag_at
        self.bu2_loss = multi_label_loss_weighted if flag_at is Flag.NOFLAG else multi_label_loss # The BU2 classification loss.
        self.task_accuracy = multi_label_accuracy_weighted if flag_at is Flag.NOFLAG else multi_label_accuracy # The task accuracy metric.
        self.project_path = Path(__file__).parents[2]
        self.train_arg_emb = False  # Whether for each task train also the argument(character) embedding.
        self.initial_tasks = [0] # The initial tasks we first train in the zero forgetting experiments.
        self.ntasks = None # The number of tasks.
        self.nclasses = None # The number of classes for each task.
        self.results_dir = None # The trained model directory.
        self.use_bu1_loss = None # Whether to use the bu1 loss.
        self.ndirections = None # The number of directions we query about.


class EmnistDataset(GenericDataset):
    def __init__(self, flag_at, ndirections = 4):
        super(EmnistDataset, self).__init__(flag_at)
        self.ntasks = 4 # As we have one language(emnist) we have one task.
        self.nclasses = [47 for _ in range(ndirections)]  # We have 47 different characters.
        self.results_dir = os.path.join(self.project_path, 'data/emnist/results')
        self.use_bu1_loss = True if flag_at is not Flag.NOFLAG else False
        self.ndirections = ndirections

class FashionmnistDataset(EmnistDataset):
    def __init__(self, flag_at, ndirections = 4):
        super(FashionmnistDataset, self).__init__(flag_at, ndirections = ndirections)
        self.nclasses = [10 for _ in range(ndirections)] # The same as emnist, just we have just 10 classes in the dataset.

class OmniglotDataset(GenericDataset):
    def __init__(self,flag_at,ndirections):
        super(OmniglotDataset, self).__init__(flag_at)
        self.initial_tasks = [27, 5, 42, 18, 33]  # The initial tasks set.
        self.ntasks = 51 # We have 51 tasks.
        self.results_dir = os.path.join(self.project_path, 'data/omniglot/results')
        raw_data_path = os.path.join(self.project_path, 'data/omniglot/RAW')
        if not os.path.exists(raw_data_path):
            raise "You didn't download the raw omniglot data."
        self.nclasses = get_omniglot_dictionary(self.initial_tasks, self.ntasks, raw_data_path) # Computing for each langauge its numbe of charatcres.
        self.use_bu1_loss = False # As there are many classes we don't use the bu1 loss.
        self.ndirections = ndirections
        self.train_arg_emb = True

class AllOptions(GenericDataset):
    def __init__(self, ds_type, flag_at, ndirections = 4):
        if ds_type is DsType.Emnist:
            self.data_obj = EmnistDataset(flag_at,ndirections = ndirections)

        if ds_type is DsType.FashionMnist:
            self.data_obj = FashionmnistDataset(flag_at, ndirections = ndirections)

        if ds_type is DsType.Omniglot:
            self.data_obj = OmniglotDataset(flag_at, ndirections = ndirections)
=======
class GenericDataset:
    def __init__(self,flag_at):
        self.bu2_loss = multi_label_loss_weighted if flag_at is Flag.NOFLAG else multi_label_loss
        self.task_accuracy = multi_label_accuracy_weighted if flag_at is Flag.NOFLAG else multi_label_accuracy
        self.flag = flag_at
        self.inputs_to_struc = None
        self.initial_tasks = None
        self.ntasks = None
        self.nclasses = None
        self.results_dir = None
        self.generelize = None
        self.use_bu1_loss = None
        self.dataset_saving_by = None
        self.ndirections = None

class EmnistDataset(GenericDataset):
    def __init__(self, flag_at, ntasks = 4):
        super(EmnistDataset, self).__init__(flag_at)
        self.initial_tasks = [0]
        self.ntasks = ntasks # TODO - CHANGE TO 1.
        self.nclasses = [47 for _ in range(ntasks)]
        self.results_dir = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/results'
        self.generelize = True if flag_at is not Flag.NOFLAG else False
        self.use_bu1_loss = True if flag_at is not Flag.NOFLAG else False
        self.dataset_saving_by = 'test'
        self.ndirections = 4 # TODO - CHANGE TO 1.

class FashionmnistDataset(EmnistDataset):
    def __init__(self, flag_at, ntasks = 4):
        super(FashionmnistDataset, self).__init__(flag_at, ntasks=4)
        self.nclasses = [10 for _ in range(ntasks)]

class OmniglotDataset(GenericDataset):
    def __init__(self,flag_at,ntasks = None):
        self.initial_tasks = [27, 5, 42, 18, 33]  # The initial tasks set.
        self.ntasks = 51
        self.results_dir = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/omniglot/results_all_omniglot'
        self.dataset_id = 'test'
        self.flag = flag_at
        raw_data_path = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/omniglot/RAW'
        if not os.path.exists(raw_data_path):
            raise "You didn't download the raw omniglot data."
        self.nclasses = get_omniglot_dictionary(self.initial_tasks, self.ntasks, raw_data_path)
        self.use_bu1_loss = False
        self.ndirections = 4
        self.generelize = False

class AllOptions(GenericDataset):
    def __init__(self,ds_type,flag_at,ntasks = 4):

        if ds_type is DsType.Emnist:
            self.data_obj = EmnistDataset(flag_at,ntasks = ntasks)

        if ds_type is DsType.FashionMnist:
            self.data_obj = FashionmnistDataset(flag_at, ntasks = ntasks)

        if ds_type is DsType.Omniglot:
            self.data_obj = OmniglotDataset(flag_at)


'''

class Model_Options_By_Flag_And_DsType:
    # Class create a struct holding the specific dataset parameters.
    def __init__(self, Flag: Flag, DsType: DsType):
        """
        Args:
            Flag: The model flag.
            DsType: The data set type.
        """
        self.Flag = Flag
        self.ds_type = DsType
        self.setup_Model()

    def setup_Model(self):
        """
        Setting up the flag options and the model options.
        """
        self.Setup_architecture_params()
        self.setup_flag()

    def Setup_architecture_params(self):
        """
        Setting up the model settings.
        """
        if self.ds_type is DsType.Omniglot:
            initial_tasks = [27, 5, 42, 18, 33]  # The initial tasks set.
            ntasks = 51
            raw_data_path = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/omniglot/RAW'
            nclasses = get_omniglot_dictionary(initial_tasks, ntasks, raw_data_path)
            results_dir = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/omniglot/results_all_omniglot'
            dataset_id = 'test'
            use_bu1_loss = False
            model_arch = 'BUTDModelShared'
            ndirections = 4
            generelize = False

        if self.ds_type is DsType.Emnist:
            initial_tasks = [0]
            ntasks = 4
            nclasses = [47 for _ in range(ntasks)]
            results_dir = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/results'
            dataset_id = 'test'
            use_bu1_loss = True
            model_arch = 'BUTDModelShared'
            generelize = True
            ndirections = 4

        if self.ds_type is DsType.FashionMnist:
            initial_tasks = [0]
            ntasks = 4
            nclasses = [10 for _ in range(ntasks)]
            results_dir = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/FashionMnist/results'
            dataset_id = 'test'
            use_bu1_loss = True
            model_arch = 'BUTDModelShared'

        if self.ds_type is DsType.Cifar10:
            initial_tasks = [0]
            ntasks = 6
            nclasses = [10 for _ in range(ntasks)]
            results_dir = '/home/sverkip/data/BU-TD/yonathan/omniglot/data/results/cifar'
            use_bu1_loss = False
            model_arch = 'BUModelSimple'

        if self.ds_type is DsType.Cifar100:
            initial_tasks = [0]
            ntasks = 6
            nclasses = [10 for _ in range(ntasks)]

        self.initial_tasks = initial_tasks
        self.ntasks = ntasks
        self.nclasses = nclasses
        self.results_dir = results_dir
        self.generelize = generelize
        self.use_bu1_loss = use_bu1_loss
        self.dataset_saving_by = dataset_id
        self.model_arch = model_arch
        self.ndirections = ndirections

    def setup_flag(self):
        """
        Setting up the flag fields.
        """
        if self.Flag is Flag.ZF:
            use_td_flag = True
            use_ZF = True
            inputs_to_struc = inputs_to_struct
            bu2_loss = multi_label_loss
            accuracy = multi_label_accuracy

        elif self.Flag is Flag.TD:
            use_td_flag = True
            use_ZF = False
            inputs_to_struc = inputs_to_struct
            bu2_loss = multi_label_loss
            accuracy = multi_label_accuracy

        elif self.Flag is Flag.NOFLAG:
            use_td_flag = False
            use_ZF = False
            inputs_to_struc = inputs_to_struct
            bu2_loss = multi_label_loss_weighted
            accuracy = multi_label_accuracy_weighted
            use_bu1_loss = False
            self.use_bu1_loss = use_bu1_loss

        self.use_td_flag = use_td_flag
        self.use_ZF = use_ZF
        self.inputs_to_struct = inputs_to_struc
        self.bu2_loss = bu2_loss
        self.task_accuracy = accuracy
        
'''
>>>>>>> 315b11ac3016dc72662fd8ca96881ae68c5cda6d
