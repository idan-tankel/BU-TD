import os
from typing import List
from datetime import datetime
from torch import nn
import yaml
import numpy as np
import yacs
from yacs.config import CfgNode as CN


class Config:
    """
     This class holds up the configuration of the Model, Losses, Datasets, Logging, Training...
     Most of the Attrs are from the config.yaml file attached

     However, some of the attrs are "computed" as default in the __init__ function and in setup_flag function, and in some other functions under Model subclass

        Attributes:
            __config: the config.yaml file as a dict
            Visibility: the Visibility Section
            RunningSpecs: the RunningSpecs Section
            Datasets: the Datasets Section
            Logging: the Logging Section
            Folders: Some saving options section
            Lossses: the Losses Section - flags to use or not the losses
            Models: the Models options, including stride, inshape, number of FC heads, etc.
            Training: the Training options, including epochs, batch size, larning rate, etc.
    """

    def __init__(self,experiment_filename="config.yaml"):
        """
        __init__ Constructs the config object from the config.yaml file. The default file is config.yaml

        Args:
            experiment_filename (str, optional): the path to the config file. This config file must have the same scheme as the original config.yaml file. Defaults to "config.yaml".
        """        
        full_path = os.path.join(os.path.dirname(__file__), experiment_filename)
        with open(full_path, 'r') as stream:
            self.__config = yaml.safe_load(stream)
        self.Visibility = Visibility(self.__config['Visibility'])
        self.RunningSpecs = RunningSpecs(self.__config['RunningSpecs'])
        self.Datasets = Datasets(self.__config['Datasets'])
        self.Logging = Logging(self.__config['Logging'])
        self.Folders = Folders(self.__config['Folders'])
        self.Strings = Strings(self.__config['strings'])
        self.Losses = Losses(self.__config['Losses'])
        self.Models = Models(self.__config['Models'])
        self.Training = Training(self.__config['Training'])
        now = datetime.now()
        dt_string = now.strftime("%d.%m.%Y %H:%M:%S")
        self.model_path = 'DS=' + self.RunningSpecs.processed_data + \
            "time = " + str(dt_string)
        self.results_dir = '../data/emnist/data/results'
        self.model_dir = os.path.join(self.results_dir, self.model_path)
        # self.setup_flag()
        # self.Data_obj = AllOptions(ds_type=DsType(self.Datasets.dataset), flag_at=Flag(self.RunningSpecs.Flag), ndirections=4)


    def setup_flag(self) -> None:
        """
        setup_flag This function set up the Architecture of the model using the FlagAt enum. The flags of the arch are attached to the Config object and later readed by the `create_model` function.

        # TODO remove similar function from the FlagAt module under supplemtery package

        Returns: None (setting attributes of the Config object `self`)
        """
        model_flag = self.RunningSpecs.Flag
        if model_flag is Flag.BU2:
            self.use_bu1_flag = False
            self.use_td_flag = False
            self.use_bu2_flag = True
        elif model_flag is Flag.BU1 or model_flag is Flag.BU1_SIMPLE or model_flag is Flag.BU1_NOFLAG:
            self.use_bu1_flag = True
            self.use_td_flag = False
            self.use_bu2_flag = False
        elif model_flag is Flag.TD:
            self.use_bu1_flag = False
            self.use_td_flag = True
            self.use_bu2_flag = False
            self.use_SF = False
        elif model_flag is Flag.SF:
            self.use_bu1_flag = False
            self.use_td_flag = True
            self.use_bu2_flag = False
            self.use_SF = True
        elif model_flag is Flag.NOFLAG:
            self.use_bu1_flag = False
            self.use_td_flag = False
            self.use_bu2_flag = False
            self.use_SF = False
        try:
            self.flag_size = self.Models.nclasses[0][0] - 1  + 4 
            # The flag size should be number of classes + 4 (number of directios \ total tasks). Since there is a background class, we have added +1
        except KeyError as e:
            print(f'The config.Models object was not initialized before calling up setup_flag {e}')
        

    def get_config(self):
        return self.__config

    def get_visibility(self):
        return self.Visibility
    
    def get_models(self):
        return self.Models


class Strings:
    def __init__(self, config: dict):
        self.features_strings: List[str] = config['features_strings']


class Models:
    def __init__(self, config: dict):
        for key, value in config.items():
            setattr(self, key, value)



class Visibility:
    def __init__(self, config: dict):
        self.interactive_session = config['interactiveSession']


class RunningSpecs:
    def __init__(self, config: dict):
        self.distributed = config['distributed']
        self.isFit = config['isFit']
        self.processed_data = config['processed_data']
        self.backbone = config['backbone']


class Datasets:
    def __init__(self, config: dict):
        # self.dataset = DsType[config['dataset']]
        self.dummyds = config['dummyds']


class Logging:
    def __init__(self, config: dict):
        for key, value in config.items():
            setattr(self, key, value)


class Losses:
    def __init__(self, config: dict):
        self.use_td_loss = config['use_td_loss']
        self.use_bu1_loss = config['use_bu1_loss']
        self.use_bu2_loss = config['use_bu2_loss']
        self.activation_fun = nn.__getattribute__(config['activation_fun'])
        # the UnifiedLossFun is a wrapper around the loss functions, and it must be the last one here since it using all the arguments of self
        # since there are not much accuracy functions written here, the multi_label_accuracy_base is hard coded most of the time


class Folders:
    def __init__(self, config: dict):
        self.data_dir = config['data_dir']
        self.avatars = config['avatars']
        self.samples = config['samples']
        self.conf = config['conf']
        self.results = config['results']


class Training:
    """
     _summary_
    """

    def __init__(self, config: dict):
        for key, value in config.items():
            self.__setattr__(key, value)
        self.lr: float = float(config['lr'])
