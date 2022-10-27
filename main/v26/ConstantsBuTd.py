import torch

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Constants:
    ''' ***DEPRECATION WARNING*** '''
    __dev = None
    __model = None
    __model_opts = None
    __inputs_to_struct = None


def get_dev():
    if Constants.__dev is None:
        Constants.__dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return Constants.__dev


def set_model(model):
    Constants.__model = model

def get_model():
    if Constants.__model is None:
        print("Model wasn't initialized!!!!!!")
    return Constants.__model


def set_model_opts(model_opts):
    Constants.__model_opts = model_opts


def get_model_opts():
    if Constants.__model_opts is None:
        print("model_opts wasn't initialized!!!!!!")
    return Constants.__model_opts


def set_inputs_to_struct(inputs_to_struct):
    Constants.__inputs_to_struct = inputs_to_struct


def get_inputs_to_struct():
    if Constants.__inputs_to_struct is None:
        print("inputs_to_struct wasn't initialized!!!!!!")
    return Constants.__inputs_to_struct
