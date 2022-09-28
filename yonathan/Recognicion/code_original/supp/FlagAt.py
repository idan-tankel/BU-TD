from enum import Enum, auto
import os
from supp.general_functions import get_omniglot_dictionary

class Flag(Enum):
    """
    The possible flags.
    """
    TD = auto()
    NOFLAG = auto()
    SF = auto()

    
class DsType(Enum):
    """
    The Data Set types.
    """
    Emnist = auto()
    FashionMnist = auto()
    Omniglot = auto()
    Cifar10 = auto()
    Cifar100 = auto()

class Model_Options_By_Flag_And_DsType:
    # Class create a struct holding the specific dataset parameters.
    def __init__(self,Flag:Flag, DsType:DsType):
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
            initial_tasks = [27, 5, 42, 18, 33] # The initial tasks set.
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
            ntasks = 1
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
