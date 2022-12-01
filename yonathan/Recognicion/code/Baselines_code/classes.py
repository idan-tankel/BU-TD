from enum import Enum

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

    def __str__(self):
        return self.value

    def class_to_reg(self, parser):
        return getattr(parser, self.__str__()+'_lambda')
