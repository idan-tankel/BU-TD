from enum import Enum, auto


class FlagAt(Enum):
    BU1 = auto()
    TD = auto()
    BU2 = auto()
    NOFLAG = auto()
    BU1_SIMPLE = auto()
    BU1_NOLAG = auto()
    SF = auto()

# TODO - ADAPT TO ALL MODELS

def setup_flag(parser):
    model_flag = parser.parse_args().model_flag

    if model_flag is FlagAt.TD:
        use_td_flag = True
        use_SF = False

    elif model_flag is FlagAt.SF:
        use_td_flag = True
        use_SF = True

    elif model_flag is FlagAt.NOFLAG:
        use_td_flag = False
        use_SF = False


    parser.add_argument('--use_td_flag', default=use_td_flag, type=staticmethod,     help='Whether to use the BU-TD model with task & arg embedding.')  #
    parser.add_argument('--use_SF', default=use_SF, type=staticmethod,       help='Whther to create task embedding at the BU2 stream')  #