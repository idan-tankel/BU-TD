"""
Constant Enums.
"""
from enum import Enum, auto


class Flag(Enum):
    """
    The possible training samples.
    """
    NOFLAG = auto()  # Non-guided model, should output for each character its adjacent neighbor.
    CL = auto()  # Continual learning flag, a BU_TD network with allocating task embedding for each task.


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


class Training_type(Enum):
    """
    Training types.
    """
    Full = 'Full'
    BU2_Modulation = 'BU2_Modulation'
    Masks = 'Masks'
    Classifier_Only = 'Classifier_Only'
    TD_Modulation = 'TD_Modulation'

    def __str__(self):
        return self.value


class DsType(Enum):
    """
    Data-Set types.
    """
    Emnist = 'Emnist'
    Fashionmnist = 'Fashionmnist'
    Omniglot = 'Omniglot'
    StanfordCars = 'StanfordCars'
    Food = 'Food'

    def __str__(self):
        return self.value


class Model_Type(Enum):
    """
    Model Types.
    """
    ResNet14 = 'ResNet14'
    ResNet18 = 'ResNet18'
    ResNet20 = 'ResNet20'
    ResNet32 = 'ResNet32'
    ResNet34 = 'ResNet34'
    ResNet50 = 'ResNet50'
    ResNet101 = 'ResNet101'
    MLP = 'MLP'
    BUTD = 'BUTD'

    def __str__(self):
        return self.value
