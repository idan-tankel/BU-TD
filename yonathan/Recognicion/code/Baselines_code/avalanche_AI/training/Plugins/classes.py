"""
Here we define for each regularization type its plugin.
"""

from Baselines_code.avalanche_AI.training.Plugins.EWC import EWC
from Baselines_code.avalanche_AI.training.Plugins.IMM_Mean import IMM_Mean as IMM_Mean
from Baselines_code.avalanche_AI.training.Plugins.IMM_Mode import MyIMM_Mode as IMM_Mode
from Baselines_code.avalanche_AI.training.Plugins.LFL import LFL
from Baselines_code.avalanche_AI.training.Plugins.LWF import LwF
from Baselines_code.avalanche_AI.training.Plugins.MAS import MAS
from Baselines_code.avalanche_AI.training.Plugins.SI import SI
from Baselines_code.avalanche_AI.training.Plugins.plugins_base import Base_plugin
import argparse

from Baselines_code.baselines_utils import RegType


def Get_regularization_plugin(opts: argparse, reg_type: RegType, prev_checkpoint: dict,
                              load_from: str) -> Base_plugin:
    """
    Returns the desired regularization plugin.
    Args:
        opts: The model model_opts.
        reg_type: Regularization type.
        prev_checkpoint: The previous model.
        load_from: load from.

    Returns: Regularization plugin.

    """
    if reg_type is RegType.EWC:
        return EWC(opts=opts, prev_checkpoint=prev_checkpoint, load_from=load_from)
    if reg_type is RegType.LFL:
        return LFL(opts, prev_checkpoint)
    if reg_type is RegType.LWF:
        return LwF(opts, prev_checkpoint)
    if reg_type is RegType.MAS:
        return MAS(opts=opts, prev_checkpoint=prev_checkpoint,
                   load_from=load_from)
    if reg_type is RegType.IMM_Mean:
        return IMM_Mean(opts=opts, prev_checkpoint=prev_checkpoint)
    if reg_type is RegType.IMM_Mode:
        return IMM_Mode(opts=opts, prev_checkpoint=prev_checkpoint,
                        load_from=load_from)
    if reg_type is RegType.SI:
        return SI(opts=opts, prev_checkpoint=prev_checkpoint)

    else:
        raise NotImplementedError
