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

    def __init__(self, in_channels: int, heads: List[int], modulation: List[List[nn.Parameter]],
                 mask, block_expansion: int):
        """
        Compute class probabilities.
        Args:
            in_channels: The in channels.
            heads: The number of heads per task.
            modulation: The modulations list.
        """
        super(Head, self).__init__()
        self.head = heads
        Heads = []
        for index, head in enumerate(heads):
            layer = nn.Linear(in_channels * block_expansion, head)
            Heads.append(layer)
            modulation[index].extend(layer.parameters())
            mask[index].extend(layer.parameters())
        Heads = nn.ModuleList(Heads)
        self.Head = Heads

    def forward(self, inputs: Tensor, task_flag: Tensor) -> Tensor:
        """
        Compute the class probabilities.
        Args:
            inputs: The model input.
            task_flag: The task flag.

        Returns:

        """
        task_flag = int(task_flag[0].argmax())
        return self.Head[task_flag](inputs)
