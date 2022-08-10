import torch
from torch import nn


class Hadamard(nn.Module):

    def __init__(self, lateral_per_neuron: bool, filters):
        nn.Module.__init__(self)
        self.lateral_per_neuron = lateral_per_neuron
        self.filters = filters
        # Create a trainable weight variable for this layer.
        if self.lateral_per_neuron:
            # not implemented....
            shape = 0  # input_shape[1:]
        else:
            shape = [self.filters, 1, 1]
        self.weights = nn.Parameter(torch.Tensor(*shape))  # define the trainable parameter
        #        nn.init.constant_(self.weights.data, 1)  # init your weights here...
        nn.init.xavier_uniform_(self.weights)  # init your weights here...

    def forward(self, inputs):
        return inputs * self.weights
