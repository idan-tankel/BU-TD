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
    AGEM = auto()
    RWALK = auto()
    Naive_with_freezing = auto()
    GEM = auto()
    IL = auto()

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
        if self == RegType.AGEM:
            return 'AGEM'
        if self == RegType.GEM:
            return "GEM"
        if self == RegType.IL:
            return "IL"

    def class_to_reg(self, parser):
        if self == RegType.EWC:
            return parser.ewc_lambda
        if self == RegType.SI:
            return parser.si_lambda
        if self == RegType.LWF:
            return parser.lwf_lambda
        if self == RegType.LFL:
            return parser.lfl_lambda
        if self == RegType.MAS:
            return parser.mas_lambda
        if self == RegType.RWALK:
            return parser.rwalk_lambda
        if self == RegType.AGEM:
            return parser.agem_nsamples
        if self == RegType.GEM:
            return parser.gem_nsamples
        if self == RegType.IL:
            return parser.IL_nsamples