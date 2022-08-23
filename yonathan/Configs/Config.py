import os
from typing import List
from datetime import datetime
import yaml

# configuration class
from v26.models.flag_at import FlagAt


class Config:
    def __init__(self):
        path = "config.yaml"
        full_path = os.path.join(os.path.dirname(__file__), path)
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
        self.model_path = 'DS=' + self.RunningSpecs.processed_data + "time = " + str(dt_string)
        self.results_dir = '../data/emnist/data/results'
        self.model_dir = os.path.join(self.results_dir, self.model_path)

        

    def get_config(self):
        return self.__config

    def get_visibility(self):
        return self.Visibility


class Strings:
    def __init__(self, config: dict):
        self.features_strings: List[str] = config['features_strings']


class Models:
    def __init__(self, config: dict):
        for key, value in config.items():
            setattr(self, key, value)
        # self.features: str = config['features']
        # self.num_heads: str = config['num_heads']


class Visibility:
    def __init__(self, config: dict):
        self.interactive_session = config['interactiveSession']


class RunningSpecs:
    def __init__(self, config: dict):
        self.distributed = config['distributed']
        self.FlagAt = FlagAt[config['FlagAt']]
        self.isFit = config['isFit']
        self.processed_data = config['processed_data']


class Datasets:
    def __init__(self, config: dict):
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


class Folders:
    def __init__(self, config: dict):
        self.data_dir = config['data_dir']
        self.avatars = config['avatars']
        self.samples = config['samples']
        self.conf = config['conf']
        self.results = config['results']



class Training:
    def __init__(self,config: dict):
        for key,value in config.items():
            self.__setattr__(key,value)