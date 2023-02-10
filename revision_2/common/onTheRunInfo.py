import numpy as np
import torch
from torch import nn


def get_loss(weight):
    return nn.CrossEntropyLoss(reduction='none', weight=weight).to(OnTheRunInfo.dev)


class OnTheRunInfo:
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    loss = nn.CrossEntropyLoss(reduction='none').to(dev)
    acc_optimum: float = 0.0
    index_hop: int = 0
    number_of_epochs_with_acc_1 = 0

    def __init__(self):
        self.last_epoch = -1
        self.optimum = -np.inf
