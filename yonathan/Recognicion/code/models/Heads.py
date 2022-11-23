import torch
from torch import nn


dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


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
    """
    OccurrenceHead FullyConeccted layer of the output features
    """

    def __init__(self, opts):
        """
        __init__ based on `nn.Linear`

        Args:
            opts (`argparse.ArgumentParser` | `Config` | `SimpleNamespace` ): Model options to determine the number of filters
        """
        super(OccurrenceHead, self).__init__()
        opts = opts.Models
        filters = opts.nclasses[0][0]  # TODO-change to support to the
        infilters = opts.nfilters[1]
        self.fc = nn.Linear(infilters, filters)
        self.activation = nn.Sigmoid()

    def forward(self, inputs):
        x = inputs
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.activation(x)
        return x


class MultiLabelHead(nn.Module):

    def __init__(self, opts):
        """
        __init__ Initialize the head and the layers with `nn.sequential` as a list of `nn.Linear` layers by the parameters
         within the opts

        Args:
            opts (`config` | `argparse.ArgumentParser` | `SimpleNamespace`): model configuration object defined in `Configs.config.py`
        """
        super(MultiLabelHead, self).__init__()
        layers = []
        opts.Models.init_model_options()
        if len(opts.Models.nclasses) == 1 or type(opts.Models.nclasses) == int:
            try:
                number_of_heads = opts.Models.number_of_linear_heads
                filters = opts.Models.nclasses[0][0]
            except AttributeError:
                number_of_heads = len(opts.Models.nclasses)
        for k in range(number_of_heads):
            if filters is None:
                filters = opts.Models.nclasses[k][0]
            infilters = opts.Models.nfilters[-1]
            k_layers = [nn.Linear(infilters, infilters), opts.Models.norm_layer(infilters, dims=1), opts.Models.activation_fun()]*(opts.Models.ntaskhead_fc - 1)
            # add last FC: plain
            k_layers += [nn.Linear(in_features=infilters,
                                   out_features=filters)]
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
            outs.append(y)
        return torch.stack(outs, dim=-1)
