import torch
from torch import nn

# from v26.funcs import instruct, init_module_weights, get_laterals
# from v26.functions.convs import conv1x1, conv2d_fun
# from v26.models.BasicBlock import BasicBlockTDLat
# from v26.models.SharedBase import BasicBlockLatShared
# from v26.models.SideAndComb import SideAndCombShared, SideAndComb


class ResNet(nn.Module):

    def __init__(self, block, opts):
        super(ResNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        stride = opts.strides[0]
        filters = opts.nfilters[0]
        inplanes = opts.inshape[0]
        self.conv1 = nn.Conv2d(inplanes, filters, kernel_size=7, stride=stride, padding=3,
                               bias=False)
        self.inplanes = filters
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        layers = []
        for k in range(1, len(opts.strides)):
            nblocks = opts.ns[k]
            stride = opts.strides[k]
            filters = opts.nfilters[k]
            layers.append(self._make_layer(
                block, filters, nblocks, stride=stride))

        self.alllayers = nn.ModuleList(layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride,
                      downsample, norm_layer, self.activation_fun))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                          norm_layer, self.activation_fun))

        return nn.ModuleList(layers)

    def forward(self, inputs):
        x = inputs
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #        x = self.maxpool(x)

        for layer in self.alllayers:
            for block in layer:
                x = block(x)

        x = self.avgpool(x)

        return x


class ResNetLatShared(nn.Module):
    __doc__=r"""
    ResNetLatShared _summary_
    """
    def __init__(self, opts, shared):
        """
        __init__ _summary_

        Args:
            opts (_type_): _description_
            shared (_type_): _description_
        """        
        super(ResNetLatShared, self).__init__()
        model_options_section = opts.Models
        self.norm_layer = model_options_section.norm_fun
        self.activation_fun = model_options_section.activation_fun
        self.inshapes = shared.inshapes
        self.use_lateral = shared.use_lateral  # incoming lateral
        filters = opts.Models.nfilters[0]
        self.use_bu1_flag = opts.use_bu1_flag
        if self.use_bu1_flag:
            # flag at BU1. It is shared across all the BU towers
            self.h_flag_bu = nn.Sequential(shared.h_flag_bu, self.norm_layer(shared.bu1_bot_neurons, dims=1),
                                           self.activation_fun())
            self.flag_shape = shared.flag_shape
            self.h_flag_bu_resized = shared.h_flag_bu_resized

        self.conv1 = nn.Sequential(
            shared.conv1, self.norm_layer(filters), self.activation_fun())
        self.bot_lat = SideAndCombShared(
            shared.bot_lat, self.norm_layer, self.activation_fun)

        layers = []
        for shared_layer in shared.alllayers:
            layers.append(self._make_layer(shared_layer))

        self.alllayers = nn.ModuleList(layers)
        self.avgpool = shared.avgpool
        if self.use_lateral:
            self.top_lat = SideAndCombShared(
                shared.top_lat, self.norm_layer, self.activation_fun)

        if not instruct(opts, 'use_top_flag'):
            use_top_flag = False
        else:
            use_top_flag = opts.use_top_flag
        self.use_top_flag = use_top_flag
        if self.use_top_flag:
            # flag at BU2. It is not shared across the BU towers
            self.top_filters = shared.top_filters
            self.h_flag_bu2 = nn.Sequential(shared.h_flag_bu2, self.norm_layer(self.top_filters, dims=1),
                                            self.activation_fun())
            self.h_top_bu2 = nn.Sequential(shared.h_top_bu2, self.norm_layer(self.top_filters, dims=1),
                                           self.activation_fun())

        # TODO: this should only be called once for all shared instances...
        init_module_weights(self.modules())

    def _make_layer(self, blocks):
        norm_layer = self.norm_layer
        layers = []
        for shared_block in blocks:
            layers.append(BasicBlockLatShared(
                shared_block, norm_layer, self.activation_fun))

        return nn.ModuleList(layers)

    def forward(self, x, flags, laterals_in):
        """
        forward the forward pass

        Args:
            x (_type_): the input tensor
            flags (_type_): 
            laterals_in (_type_): _description_

        Returns:
            _type_: _description_
        """        
        if self.use_bu1_flag:
            f = self.h_flag_bu(flags)
            f = f.view(self.flag_shape)
            f = self.h_flag_bu_resized(f)
            x = torch.cat((x, f), dim=1)

        laterals_out = []
        x = self.conv1(x)
        lateral_layer_id = 0
        lateral_in = get_laterals(laterals_in, lateral_layer_id, None)
        if lateral_in is not None:
            x = self.bot_lat((x, lateral_in))
        laterals_out.append(x)

        for layer_id, layer in enumerate(self.alllayers):
            layer_lats_out = []
            for block_id, block in enumerate(layer):
                lateral_layer_id = layer_id + 1
                cur_lat_in = get_laterals(
                    laterals_in, lateral_layer_id, block_id)
                x, block_lats_out = block((x, cur_lat_in))
                layer_lats_out.append(block_lats_out)

            laterals_out.append(layer_lats_out)

        x = self.avgpool(x)
        lateral_in = get_laterals(laterals_in, lateral_layer_id + 1, None)
        if self.use_lateral and lateral_in is not None:
            x = self.top_lat((x, lateral_in))

        if self.use_top_flag:
            flag_bu2 = self.h_flag_bu2(flags)
            flag_bu2 = flag_bu2.view((-1, self.top_filters, 1, 1))
            x = torch.cat((x, flag_bu2), dim=1)
            x = torch.flatten(x, 1)
            x = self.h_top_bu2(x)
            x = x.view((-1, self.top_filters, 1, 1))

        laterals_out.append(x)

        return x, laterals_out

class ResNetTDLat(nn.Module):
    """ResNetTDLat is ResNet network with lateral connections
    """

    def __init__(self, opts):
        super(ResNetTDLat, self).__init__()
        block = BasicBlockTDLat
        self.use_lateral = opts.Models.use_lateral_butd
        self.activation_fun = opts.Models.activation_fun
        self.use_td_flag = opts.use_td_flag
        self.norm_layer = opts.Models.norm_fun

        top_filters = opts.Models.nfilters[-1]
        self.top_filters = top_filters
        self.inplanes = top_filters
        # TODO flag_size is missing
        # When training on emnist flag_size should be `flag_size = ndirections + nclasses_existence` 
        # When training on avatar flag_size should be `flag_size = nclasses_existence + nfeatures`
        if opts.use_td_flag:
            self.h_flag_td = nn.Sequential(nn.Linear(opts.flag_size, top_filters), self.norm_layer(top_filters, dims=1),
                                           self.activation_fun())
            self.h_top_td = nn.Sequential(nn.Linear(top_filters * 2, top_filters), self.norm_layer(top_filters, dims=1),
                                          self.activation_fun())

        upsample_size = opts.avg_pool_size  # before avg pool we have 7x7x512
        self.top_upsample = nn.Upsample(
            scale_factor=upsample_size, mode='bilinear', align_corners=False)
        #        if self.use_lateral:
        #           self.top_lat = SideAndComb(lateral_per_neuron=False,filters=top_filters)

        layers = []
        for k in range(len(opts.Models.strides) - 1, 0, -1):
            nblocks = opts.Models.ns[k]
            stride = opts.Models.strides[k]
            filters = opts.Models.nfilters[k - 1]
            layers.append(self._make_layer(
                block, filters, nblocks, stride=stride))

        self.alllayers = nn.ModuleList(layers)
        filters = opts.Models.nfilters[0]
        if self.use_lateral:
            self.bot_lat = SideAndComb(
                False, filters, self.norm_layer, self.activation_fun)
        self.use_final_conv = opts.Models.use_final_conv
        if self.use_final_conv:
            # here we should have performed another convolution to match BU conv1, but
            # we don't, as that was the behaviour in TF. Unless use_final_conv=True
            conv1 = conv2d_fun(filters, filters, kernel_size=7, stride=1, padding=3,
                               bias=False)
            self.conv1 = nn.Sequential(
                conv1, self.norm_layer(filters), self.activation_fun())

        init_module_weights(self.modules())

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self.norm_layer
        layers = []
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes, 1,
                          norm_layer, self.activation_fun, self.use_lateral))
        layers.append(block(self.inplanes, planes, stride,
                      norm_layer, self.activation_fun, self.use_lateral))
        self.inplanes = planes * block.expansion

        return nn.ModuleList(layers)

    def forward(self, bu_out,flag,laterals_in):
        #TODO change input model to **kwargs or to fix the inputs!
        # Get rid of the `inputs` variable
        # bu_out, flag, laterals_in = inputs
        laterals_out = []

        if self.use_td_flag:
            top_td = self.h_flag_td(flag)
            top_td = top_td.view((-1, self.top_filters, 1, 1))
            top_td_embed = top_td
            h_side_top_td = bu_out
            top_td = torch.cat((h_side_top_td, top_td), dim=1)
            top_td = torch.flatten(top_td, 1)
            top_td = self.h_top_td(top_td)
            top_td = top_td.view((-1, self.top_filters, 1, 1))
            x = top_td
        else:
            x = bu_out

        laterals_out.append(x)

        x = self.top_upsample(x)
        if laterals_in is None or not self.use_lateral:
            for layer in self.alllayers:
                layer_lats_out = []
                for block in layer:
                    x, block_lats_out = block((x, None))
                    layer_lats_out.append(block_lats_out)

                reverse_layer_lats_out = layer_lats_out[::-1]
                laterals_out.append(reverse_layer_lats_out)
        else:
            reverse_laterals_in = laterals_in[::-1]

            for layer, lateral_in in zip(self.alllayers, reverse_laterals_in[1:-1]):
                layer_lats_out = []
                reverse_lateral_in = lateral_in[::-1]
                for block, cur_lat_in in zip(layer, reverse_lateral_in):
                    reverse_cur_lat_in = cur_lat_in[::-1]
                    x, block_lats_out = block((x, reverse_cur_lat_in))
                    layer_lats_out.append(block_lats_out)

                reverse_layer_lats_out = layer_lats_out[::-1]
                laterals_out.append(reverse_layer_lats_out)

            lateral_in = reverse_laterals_in[-1]
            x = self.bot_lat((x, lateral_in))

        if self.use_final_conv:
            x = self.conv1(x)
        laterals_out.append(x)

        outs = [x, laterals_out[::-1]]
        if self.use_td_flag:
            outs += [top_td_embed, top_td]
        return outs
