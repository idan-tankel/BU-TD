import torch
from torch import nn

from v26.functions.convs import conv3x3up

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class ImageHead(nn.Module):

    def __init__(self, opts):
        super(ImageHead, self).__init__()
        image_planes = opts.inshape[0]
        upsample_size = opts.strides[0]
        infilters = opts.nfilters[0]
        self.conv = conv3x3up(infilters, image_planes, upsample_size)

    def forward(self, inputs):
        x = self.conv(inputs)
        return x


class OccurrenceHead(nn.Module):

    def __init__(self, opts):
        super(OccurrenceHead, self).__init__()
        filters = opts.nclasses_existence
        infilters = opts.nfilters[-1]
        self.fc = nn.Linear(infilters, filters)

    def forward(self, inputs):
        x = inputs
        x = torch.flatten(x, 1)
        x = self.fc(x)
        #        x = nn.Sigmoid()(x)
        return x


class MultiLabelHead(nn.Module):

    def __init__(self, opts):
        super(MultiLabelHead, self).__init__()
        layers = []
        for k in range(len(opts.Models.nclasses)):
            filters = opts.Models.nclasses[k][0]
            k_layers = []
            infilters = opts.Models.nfilters[-1]
            for i in range(opts.Models.ntaskhead_fc - 1):
                k_layers += [nn.Linear(infilters, infilters), opts.norm_fun(infilters, dims=1), opts.activation_fun()]

            # add last FC: plain
            k_layers += [nn.Linear(in_features=infilters, out_features=filters)]
            if len(k_layers) > 1:
                k_layers = nn.Sequential(*k_layers)
            else:
                k_layers = k_layers[0]
            layers.append(k_layers)

        self.layers = nn.ModuleList(layers)

    def forward(self, inputs, flags=None):
        x = inputs
        x = torch.flatten(x, 1)
        outs = []
        for layer in self.layers:
            y = layer(x)
            outs.append(y)
        return torch.stack(outs, dim=-1)  # Here output of: taskhead


class MultiLabelHeadOnlyTask(MultiLabelHead):
    def forward(self, inputs, flags=None):
        x = inputs
        x = torch.flatten(x, 1)
        outs = []
        for index, layer in enumerate(self.layers):
            y = layer(x)
            # TODO itsik!!!!Here should get the task - and check if should activate loss - for each one of images in the batch
            #  checkfor imgae - the flag - and append it if relevant

            outs.append(y)
        return torch.stack(outs, dim=-1)  # Here output of: taskhead
