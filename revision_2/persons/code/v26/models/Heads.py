import torch
from torch import nn

from persons.code.v26.functions.convs import conv3x3up


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
        for k in range(len(opts.nclasses)):
            filters = opts.nclasses[k]
            k_layers = []
            infilters = opts.nfilters[-1]
            for i in range(opts.ntaskhead_fc - 1):
                k_layers += [nn.Linear(infilters, infilters), opts.norm_fun(infilters, dims=1), opts.activation_fun()]

            # add last FC: plain
            k_layers += [nn.Linear(infilters, filters)]
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
        num_persons_start_features = 6  # TODO - 6 - to will be the number of features

        for curr_example in list(range(x.shape[0])):
            example_output = []
            task_x = x[curr_example, :]
            flags_layers = [idx for idx, val in enumerate(flags[curr_example, num_persons_start_features:] == 1) if val]
            # TODO - create function for extracting the flag
            # flags_layers = torch.where(flags[curr_example][num_persons_start_features:] == 1)
            for curr_layer in flags_layers:
                example_layer_output = self.layers[curr_layer](task_x)
                example_output.append(example_layer_output)
            outs.append(example_output)
        # for index, layer in enumerate(self.layers):
        #     is_in_curr_task = flags[:, num_persons_start_features + index] == 1
        #     task_x = x[is_in_curr_task, :]
        #     y = layer(task_x)
        #     outs.append(y)
        return outs
        # return torch.stack(outs, dim=-1)  # Here output of: taskhead

