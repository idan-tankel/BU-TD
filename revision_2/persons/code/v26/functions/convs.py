from torch import nn


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size, stride=1, padding=1, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, stride=stride, padding=padding, groups=nin,
                                   bias=False)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

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
