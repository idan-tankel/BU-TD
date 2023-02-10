import numpy as np
from torch import nn

from v26.funcs import orig_relus
from v26.functions.convs import conv3x3, conv1x1, conv2d_fun
from v26.models.SideAndComb import SideAndCombSharedBase, SideAndCombShared


class BasicBlockLatSharedBase():
    expansion = 1

    def __init__(self, inplanes, planes, stride, use_lateral):
        super(BasicBlockLatSharedBase, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        downsample = None
        if stride != 1 or inplanes != planes * BasicBlockLatSharedBase.expansion:
            downsample = conv1x1(inplanes, planes * BasicBlockLatSharedBase.expansion, stride)
        self.downsample = downsample
        self.stride = stride
        self.use_lateral = use_lateral
        if self.use_lateral:
            self.lat1 = SideAndCombSharedBase(lateral_per_neuron=False, filters=inplanes)
            self.lat2 = SideAndCombSharedBase(lateral_per_neuron=False, filters=planes)
            self.lat3 = SideAndCombSharedBase(lateral_per_neuron=False, filters=planes)
        self.inplanes = inplanes
        self.planes = planes


class BasicBlockLatShared(nn.Module):

    def __init__(self, shared, norm_layer, activation_fun):
        super(BasicBlockLatShared, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        planes = shared.planes
        self.conv1 = nn.Sequential(shared.conv1, norm_layer(planes), activation_fun())
        self.conv2 = nn.Sequential(shared.conv2, norm_layer(planes), activation_fun())
        if shared.downsample is not None:
            downsample = nn.Sequential(shared.downsample, norm_layer(planes))
        else:
            downsample = None
        self.downsample = downsample
        self.stride = shared.stride
        self.use_lateral = shared.use_lateral
        if self.use_lateral:
            self.lat1 = SideAndCombShared(shared.lat1, norm_layer, activation_fun)
            self.lat2 = SideAndCombShared(shared.lat2, norm_layer, activation_fun)
            self.lat3 = SideAndCombShared(shared.lat3, norm_layer, activation_fun)
        if orig_relus:
            self.relu = activation_fun()

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
        if laterals_in is not None:
            x = self.lat2((x, lateral2_in))
        laterals_out.append(x)

        x = self.conv2(x)
        if laterals_in is not None:
            x = self.lat3((x, lateral3_in))
        laterals_out.append(x)

        if self.downsample is not None:
            identity = self.downsample(inp)
        else:
            identity = inp

        x = x + identity
        if orig_relus:
            x = self.relu(x)

        return x, laterals_out


class ResNetLatSharedBase():

    def __init__(self, opts):
        super(ResNetLatSharedBase, self).__init__()
        self.activation_fun = opts.activation_fun
        self.use_lateral = opts.use_lateral_tdbu  # incoming lateral
        stride = opts.strides[0]
        filters = opts.nfilters[0]
        inplanes = opts.inshape[0]
        inshape = np.array(opts.inshape)
        self.use_bu1_flag = opts.use_bu1_flag
        if self.use_bu1_flag:
            lastAdded_shape = opts.inshape
            flag_scale = 2
            self.flag_shape = [-1, 1, lastAdded_shape[1] // flag_scale, lastAdded_shape[2] // flag_scale]
            bu1_bot_neurons = int(np.product(self.flag_shape[1:]))
            self.bu1_bot_neurons = bu1_bot_neurons
            self.h_flag_bu = nn.Linear(opts.flag_size, bu1_bot_neurons)
            self.h_flag_bu_resized = nn.Upsample(scale_factor=flag_scale, mode='bilinear', align_corners=False)
            inplanes += 1

        inshapes = []
        self.conv1 = conv2d_fun(inplanes, filters, kernel_size=7, stride=stride, padding=3,
                                bias=False)
        self.inplanes = filters
        inshape = np.array([filters, inshape[1] // stride, inshape[2] // stride])
        inshapes.append(inshape)
        # for BU2 use the final TD output as an input lateral. Note that this should be done even when not using laterals
        self.bot_lat = SideAndCombSharedBase(lateral_per_neuron=False, filters=filters)

        layers = []  # groups. each group has n blocks
        for k in range(1, len(opts.strides)):
            nblocks = opts.ns[k]
            stride = opts.strides[k]
            filters = opts.nfilters[k]
            layers.append(self._make_layer(filters, nblocks, stride=stride))
            inshape = np.array([filters, inshape[1] // stride, inshape[2] // stride])
            inshape_lst = []
            for _ in range(nblocks):
                inshape_lst.append(inshape)
            inshapes.append(inshape_lst)

        self.alllayers = layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        filters = opts.nfilters[-1]
        if self.use_lateral:
            self.top_lat = SideAndCombSharedBase(lateral_per_neuron=False, filters=filters)
        inshape = np.array([filters, 1, 1])
        inshapes.append(inshape)
        self.inshapes = inshapes

        self.use_bu2_flag = opts.use_bu2_flag
        if self.use_bu2_flag:
            top_filters = opts.nfilters[k]
            self.top_filters = top_filters
            self.h_flag_bu2 = nn.Linear(opts.flag_size, top_filters)
            self.h_top_bu2 = nn.Linear(top_filters * 2, top_filters)

    def _make_layer(self, planes, blocks, stride=1):
        layers = []
        layers.append(BasicBlockLatSharedBase(self.inplanes, planes, stride, self.use_lateral))
        self.inplanes = planes * BasicBlockLatSharedBase.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlockLatSharedBase(self.inplanes, planes, 1, self.use_lateral))

        return layers
