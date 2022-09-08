from torch import nn
import torch


# TODO add this as a method to model / training / another part of the code!
def get_model_outs(model: nn.Module, outs: list[torch]) -> object:
    """
    :param model: The model
    :param outs: The list of outputs of the model from all the streams.
    :return: struct containing all tensor in the list
    """
    if type(model) is torch.nn.DataParallel or type(model) is torch.nn.parallel.DistributedDataParallel:
        return model.module.outs_to_struct(outs)  # Use outs_to_struct to transform from list -> struct
    else:
        return model.outs_to_struct(outs)