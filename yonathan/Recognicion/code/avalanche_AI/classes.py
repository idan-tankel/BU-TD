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
<<<<<<< HEAD
    RWALK = auto()
=======
>>>>>>> 315b11ac3016dc72662fd8ca96881ae68c5cda6d


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
<<<<<<< HEAD
        if self == RegType.RWALK:
            return 'RWALK'
=======
>>>>>>> 315b11ac3016dc72662fd8ca96881ae68c5cda6d



