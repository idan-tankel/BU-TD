from enum import Enum, auto

class RegType(Enum):
    """
    The possible baselines.
    """
    EWC = auto()
    SI = auto()
    LWF = auto()
    LFL = auto()