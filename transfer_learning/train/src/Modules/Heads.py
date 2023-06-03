"""
Model Head part.
"""
import torch.nn as nn

from typing import List

from torch import Tensor


class Head(nn.Module):
    """
    Head class.
    Supports head per task.
    """

    def __init__(self, opts, in_channels: int, modulation: List[List[nn.Parameter]],
                 mask: list):
        """
        Compute class probabilities.
        Args:
            opts: The argument parser.
            in_channels: The in channels.
            modulation: The modulations list.
            mask: The mask weights.

        """
        super(Head, self).__init__()
        self.heads = opts.heads
        Heads = []
        for index, head in enumerate(self.heads):
            layer = nn.Linear(in_channels, head)
            Heads.append(layer)
            if modulation is not None:
                modulation[index].extend(layer.parameters())
                mask[index].extend(layer.parameters())
        Heads = nn.ModuleList(Heads)
        self.Head = Heads

    def forward(self, inputs: Tensor, task_flag: Tensor) -> Tensor:
        """
        Compute the class probabilities.
        Args:
            inputs: The model x.
            task_flag: The task flag.

        Returns:

        """
        task_flag = task_flag[0]
        return self.Head[task_flag](inputs)
