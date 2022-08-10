from torch import nn

from v26.funcs import orig_relus
from v26.models.Hadamard import Hadamard


class SideAndCombSharedBase():
    """
    SideAndCombSharedBase _summary_
    """    
    def __init__(self, lateral_per_neuron=False, filters):
        """
        __init__ _summary_

        Args:
            lateral_per_neuron (bool): 
            filters (_type_): _description_
        """ 
        # super(SideAndCombSharedBase, self).__init__()
        self.side = Hadamard(lateral_per_neuron, filters)
        self.filters = filters


class SideAndCombShared(nn.Module):
    """
    SideAndCombShared _summary_

    Args:
        nn (_type_): _description_
    """    
    def __init__(self, shared, norm_layer, activation_fun):
        """
        __init__ _summary_

        Args:
            shared (_type_): _description_
            norm_layer (_type_): _description_
            activation_fun (_type_): _description_
        """        
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
    """
    SideAndComb _summary_
    """    
    def __init__(self, lateral_per_neuron: int, filters, norm_layer, activation_fun):
        """
        __init__ _summary_

        Args:
            lateral_per_neuron (int): _description_
            filters (_type_): _description_
            norm_layer (_type_): _description_
            activation_fun (_type_): _description_
        """        
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
