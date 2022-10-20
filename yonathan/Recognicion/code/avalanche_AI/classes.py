from enum import Enum, auto

class RegType(Enum):
    """
    The possible baselines.
    """
    EWC = auto()
    SI = auto()
    LWF = auto()
    LFL = auto()
    MAS = auto()
    RWALK = auto()


    def Enum_to_name(self):
        if self == RegType.EWC:
            return 'EWC'
        if self == RegType.SI:
            return 'SI'
        if self == RegType.LWF:
            return 'LWF'
        if self == RegType.LFL:
            return 'LFL'
        if self == RegType.MAS:
            return 'MAS'
        if self == RegType.RWALK:
            return 'RWALK'



