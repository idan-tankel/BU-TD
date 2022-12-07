from enum import Enum, auto


class FlagAt(Enum):
    """
    FlagAt _summary_

    Args:
        Enum (_type_): _description_
    """    
    BU1 = auto()
    TD = auto()
    BU2 = auto()
    NOFLAG = auto()
    BU1_SIMPLE = auto()
    BU1_NOLAG = auto()
    SF = auto()


def setup_flag(parser):
    """
    setup_flag _summary_

    Args:
        parser ( `argparse` | `SimpleNamespace`): The options / parser object
    """    
    model_flag = parser.parse_args().model_flag
    if model_flag is FlagAt.BU2:
        use_bu1_flag = False
        use_td_flag = False
        use_bu2_flag = True
    elif model_flag is FlagAt.BU1 or model_flag is FlagAt.BU1_SIMPLE or model_flag is FlagAt.BU1_NOLAG:
        use_bu1_flag = True
        use_td_flag = False
        use_bu2_flag = False
    elif model_flag is FlagAt.TD:
        use_bu1_flag = False
        use_td_flag = True
        use_bu2_flag = False
        use_SF = False
    elif model_flag is FlagAt.SF:
        use_bu1_flag = False
        use_td_flag = True
        use_bu2_flag = False
        use_SF = True
    elif model_flag is FlagAt.NOFLAG:
        use_bu1_flag = False
        use_td_flag = False
        use_bu2_flag = False
        use_SF = False

    # TODO change that since this function is only setting up the parser!!!
    parser.add_argument('--use_bu1_flag', default=use_bu1_flag, type=staticmethod,
                        help='The unified loss function of all training')  #
    parser.add_argument('--use_td_flag', default=use_td_flag, type=staticmethod,
                        help='The unified loss function of all training')  #
    parser.add_argument('--use_bu2_flag', default=use_bu2_flag, type=staticmethod,
                        help='The unified loss function of all training')  #
    parser.add_argument('--use_SF', default=use_SF, type=staticmethod,
                        help='The unified loss function of all training')  #
