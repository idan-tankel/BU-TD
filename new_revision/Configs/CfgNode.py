from yacs import config
from yacs.config import CfgNode as CN
config._VALID_TYPES.add(type(None))
_config = CN()
_config.Visibility = CN()
_config.Visibility.interactiveSession = True
_config.RunningSpecs = CN()
_config.Datasets = CN()
_config.Logging = CN()
_config.Folders = CN()
_config.Losses = CN()
_config.Models = CN()
_config.Training = CN()
_config.merge_from_file("new_revision/Configs/config.yaml")
_config.freeze()
print(_config)
