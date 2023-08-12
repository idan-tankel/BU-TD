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

    def __init__(self, in_channels: int, heads: List[int], drop_out_rate = .2):
        """
        Compute class probabilities.
        Args:
            in_channels: The in channels.
            heads: The number of heads per task.
            modulation: The modulations list.
        """
        super(Head, self).__init__()
        self.head = heads
        self.drop_out = nn.Dropout(p=drop_out_rate)
        Heads = []
        for index, head in enumerate(heads):
            layer = nn.Linear(in_channels, head)
            Heads.append(layer)
        Heads = nn.ModuleList(Heads)
        self.Head = Heads

    def forward(self, inputs: Tensor, task: Tensor) -> Tensor:
        """
        Compute the class probabilities.
        Args:
            inputs: The model x.
            task: The task flag.

        Returns:

        """
        task = task[0]
        inputs = self.drop_out(inputs)
        return self.Head[task](inputs)
