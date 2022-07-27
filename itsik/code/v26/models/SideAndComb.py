from torch import nn

from persons.code.v26.funcs import orig_relus
from persons.code.v26.models.Hadamard import Hadamard


class SideAndCombSharedBase():
    def __init__(self, lateral_per_neuron, filters):
        super(SideAndCombSharedBase, self).__init__()
        self.side = Hadamard(lateral_per_neuron, filters)
        self.filters = filters


class SideAndCombShared(nn.Module):
    def __init__(self, shared, norm_layer, activation_fun):
        super(SideAndCombShared, self).__init__()
        self.side = shared.side
        self.norm = norm_layer(shared.filters)
        if not orig_relus:
            self.relu1 = activation_fun()
        self.relu2 = activation_fun()

    def forward(self, inputs):
        x, lateral = inputs

        side_val = self.side(lateral)
        side_val = self.norm(side_val)
        if not orig_relus:
            side_val = self.relu1(side_val)
        x = x + side_val
        x = self.relu2(x)
        return x


class SideAndComb(nn.Module):
    def __init__(self, lateral_per_neuron, filters, norm_layer, activation_fun):
        super(SideAndComb, self).__init__()
        self.side = Hadamard(lateral_per_neuron, filters)
        self.norm = norm_layer(filters)
        if not orig_relus:
            self.relu1 = activation_fun()
        self.relu2 = activation_fun()

    def forward(self, inputs):
        x, lateral = inputs

        side_val = self.side(lateral)
        side_val = self.norm(side_val)
        if not orig_relus:
            side_val = self.relu1(side_val)
        x = x + side_val
        x = self.relu2(x)
        return x
