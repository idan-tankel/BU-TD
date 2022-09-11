from enum import Enum, auto
import os
def folder_size(path: str) -> int:
    """

    :param path:path to a language file.
    :return: number of characters in the language
    """
    return len([_ for _ in os.scandir(path)])


def create_dict(path: str) -> dict:
    """

    :param path: path to all raw Omniglot languages
    :return: dictionary of number of characters per language
    """
    dict_language = {}
    cnt = 0
    for ele in os.scandir(path):  # for language in Omniglot_raw find the number of characters in it.
        dict_language[cnt] = folder_size(ele)  # Find number of characters in the folder.
        cnt += 1
    return dict_language


def get_omniglot_dictionary(initial_tasks, ntasks, raw_data_folderpath: str) -> list:
    """
    :param opts:
    :param raw_data_folderpath:
    :return: list of nclasses per language.
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

class Flag(Enum):
    TD = auto()
    NOFLAG = auto()
    SF = auto()
    
class DsType(Enum):
    Emnist = auto()
    FashionMnist = auto()
    Omniglot = auto()
    Cifar10 = auto()
    Cifar100 = auto()

class Model_Options_By_Flag_And_DsType:
    def __init__(self,Flag, DsType):
        self.Flag = Flag
        self.ds_type = DsType
        self.setup_Model()

    def setup_Model(self):
        self.Setup_architecture_params()
        self.setup_flag()

    def Setup_architecture_params(self):
        if self.ds_type is DsType.Omniglot:
            initial_tasks = [27, 5, 42, 18, 33]
            ntasks = 51
            raw_data_path = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/omniglot/RAW'
            nclasses = get_omniglot_dictionary(initial_tasks, ntasks, raw_data_path)
            results_dir = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/omniglot/results'
            dataset_id = 'test'
            use_bu1_loss = False
            model_arch = 'BUTDModelShared'

        if self.ds_type is DsType.Emnist:
            initial_tasks = [0]
            ntasks = 4
            nclasses = [47 for _ in range(ntasks)]
            results_dir = '/home/sverkip/data/BU-TD/yonathan/omniglot/data/results/emnist'
            dataset_id = 'val'
            use_bu1_loss = True
            model_arch = 'BUTDModelShared'

        if self.ds_type is DsType.FashionMnist:
            initial_tasks = [0]
            ntasks = 4
            nclasses = [10 for _ in range(ntasks)]
            results_dir = '/home/sverkip/data/BU-TD/yonathan/omniglot/data/results/FashionMnist'
            dataset_id = 'val'
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
        self.use_bu1_loss = use_bu1_loss
        self.dataset_id = dataset_id
        self.model_arch = model_arch

    def setup_flag(self):
        if self.Flag is Flag.SF:
            use_td_flag = True
            use_SF = True

        elif self.Flag is Flag.TD:
            use_td_flag = True
            use_SF = False


        elif self.Flag is Flag.NOFLAG:
            use_td_flag = False
            use_SF = False

        self.use_td_flag = use_td_flag
        self.use_SF = use_SF


