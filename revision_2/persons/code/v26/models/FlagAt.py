from enum import Enum, auto


class FlagAt(Enum):
    BU1 = auto()
    TD = auto()
    BU2 = auto()
    NOFLAG = auto()
    BU1_SIMPLE = auto()
    BU1_NOLAT = auto()

    @staticmethod
    def from_str(label):
        if label == 'BU1':
            return FlagAt.BU1
        elif label == 'TD':
            return FlagAt.TD
        elif label == 'BU2':
            return FlagAt.BU2
        elif label == 'NOFLAG':
            return FlagAt.NOFLAG
        elif label == 'BU1_SIMPLE':
            return FlagAt.BU1_SIMPLE
        elif label == 'BU1_NOLAT':
            return FlagAt.BU1_NOLAT
        else:
            raise ValueError('Unknown FlagAt label: ' + label)
