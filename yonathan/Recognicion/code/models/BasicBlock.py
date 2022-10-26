from torch import nn

# from v26.functions.convs import conv3x3, conv3x3up, conv1x1
# from v26.models.SideAndComb import SideAndComb
from models.SideAndComb import SideAndComb
from supp.general_functions import conv1x1, conv3x3, conv3x3up


class BasicBlockTDLat(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, norm_layer, activation_fun, use_lateral, orig_relus):
        super(BasicBlockTDLat, self).__init__()
        if use_lateral:
            self.lat1 = SideAndComb(
                False, inplanes, norm_layer, activation_fun, orig_relus=orig_relus)
            self.lat2 = SideAndComb(
                False, inplanes, norm_layer, activation_fun, orig_relus=orig_relus)
            self.lat3 = SideAndComb(
                False, planes, norm_layer, activation_fun, orig_relus=orig_relus)
        self.conv1 = nn.Sequential(
            conv3x3(inplanes, inplanes), norm_layer(inplanes))
        self.relu1 = activation_fun()
        self.conv2 = nn.Sequential(
            conv3x3up(inplanes, planes, stride), norm_layer(planes))
        if orig_relus:
            self.relu2 = activation_fun()
        self.relu3 = activation_fun()
        upsample = None
        outplanes = planes * BasicBlockTDLat.expansion
        if stride != 1:
            upsample = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode='bilinear',
                            align_corners=False),
                conv1x1(inplanes, outplanes, stride=1),
                norm_layer(outplanes)
            )
        elif inplanes != outplanes:
            upsample = nn.Sequential(
                conv1x1(inplanes, outplanes, stride=1),
                norm_layer(outplanes)
            )
        self.upsample = upsample
        self.stride = stride

    def forward(self, inputs):
        x, laterals_in = inputs
        if laterals_in is not None:
            lateral1_in, lateral2_in, lateral3_in = laterals_in
        laterals_out = []

        if laterals_in is not None:
            x = self.lat1((x, lateral1_in))
        laterals_out.append(x)
        inp = x

        x = self.conv1(x)
        x = self.relu1(x)
        if laterals_in is not None:
            x = self.lat2((x, lateral2_in))
        laterals_out.append(x)

        x = self.conv2(x)
        if orig_relus:
            x = self.relu2(x)
        if laterals_in is not None:
            x = self.lat3((x, lateral3_in))
        laterals_out.append(x)

        if self.upsample is not None:
            identity = self.upsample(inp)
        else:
            identity = inp

        x = x + identity
        x = self.relu3(x)

        return x, laterals_out[::-1]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, inputs):
        x = inputs

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        out += identity
        out = self.relu(out)

        return out
