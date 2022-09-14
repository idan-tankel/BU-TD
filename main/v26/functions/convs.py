from typing import Tuple
from torch import nn
from torch import device,cuda
dev = device("cuda") if cuda.is_available() else device("cpu")

class depthwise_separable_conv(nn.Module):
    """
    depthwise_separable_conv _summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, nin: int, nout: int, kernel_size: Tuple, stride=1, padding=1, bias=False):
        """
        __init__ 

        Args:
            nin (int): the number of input channels
            nout (int): the number of output channels
            kernel_size (Tuple): 
            stride (int, optional): _description_. Defaults to 1.
            padding (int, optional): `Conv2d` padding. Defaults to 1.
            bias (bool, optional): . Defaults to False.
        """
        # TODO change this to inherit from conv (?)
        nn.Module.__init__(self)
        self.depthwise = nn.Conv2d(in_channels=nin, out_channels=nin, kernel_size=kernel_size, stride=stride, padding=padding, groups=nin,
                                   bias=False)
        self.pointwise = nn.Conv2d(
            in_channels=nin, out_channels=nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


# conv2d_fun = nn.Conv2d
conv2d_fun = depthwise_separable_conv


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution"""
    return conv2d_fun(in_planes, out_planes, kernel_size=3, stride=stride,
                      padding=1, bias=False)


def conv3x3up(in_planes, out_planes, upsample_size=1):
    """upsample then 3x3 convolution"""
    layer = conv3x3(in_planes, out_planes)
    if upsample_size > 1:
        layer = nn.Sequential(nn.Upsample(scale_factor=upsample_size, mode='bilinear', align_corners=False),
                              layer)
    return layer


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
