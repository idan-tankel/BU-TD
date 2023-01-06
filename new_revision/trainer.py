import pytorch_lightning as pl
from torch import load
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path
from transformers import ViTForImageClassification, ViTConfig, TrainingArguments
from torchvision import datasets, transforms
from models.create_model import ModelWrapper
from torch.utils.data import DataLoader
from torch import cuda, device
from Configs.Config import Config
from types import SimpleNamespace
from create_dataset.datasets import DatasetAllDataSetTypesAll
import git
import argparse
import os
parser = argparse.ArgumentParser(description="conf file path to run net")
parser.add_argument('-f', '--file') 
git_repo = git.Repo(os.getcwd(), search_parent_directories=True)
git_root = Path(git_repo.working_dir)
results_dir = rf"{git_root.parent}/data/emnist/results"
os.makedirs(results_dir, exist_ok=True)
data_dir = Path(rf"{git_root.parent}/data")
os.makedirs(data_dir, exist_ok=True)
checkpoints_dir = rf"{git_root.parent}/data/emnist/checkpoints/good"
os.makedirs(results_dir, exist_ok=True)
args = parser.parse_args()
try:
    file = args.config
except AttributeError:
    file =  "small_vit.yaml"

####################### get global config ######################
global_config = Config(experiment_filename=file)
# transform = [transforms.ToTensor()]


####################### Dataset loading ########################
transform= None
compatibility_dataset = DatasetAllDataSetTypesAll(root=rf'/home/idanta/data/6_extended_testing/train/', opts=global_config,  direction=1,
                                                  is_train=True, obj_per_row=6, obj_per_col=1, split=False,nexamples=100000,transforms=transform)
compatibility_dl = DataLoader(compatibility_dataset, batch_size=global_config.Training.batch_size,
                              num_workers=global_config.Training.num_workers)
wandb_logger = WandbLogger(project="BU_TD",
                           job_type='train', log_model=True, save_dir=results_dir)
wandb_checkpoints = ModelCheckpoint(
    dirpath=checkpoints_dir, monitor="train_loss_epoch", mode="min")



####################### Model building #########################
if global_config.Models.pretrained_model_name is not None and "local" not in str(global_config.Models.pretrained_model_name):
        model = ViTForImageClassification.from_pretrained(
        global_config.Models.pretrained_model_name)
    # else create the model according to the config params
# local config is none - load empty model
else:
    model = ViTForImageClassification(
        ViTConfig(
            image_size=global_config.Models.image_size,
            hidden_size=768,
            num_hidden_layers=global_config.Models.depth,
            num_classes=global_config.Models.num_classes,
            qkv_bias=False,
            num_attention_heads=global_config.Models.num_attention_heads
        )
    )
model = ModelWrapper(model=model, config=global_config)
if "local" in str(global_config.Models.pretrained_model_name): #load from huggingface
    old_model = load(f"{checkpoints_dir}/{global_config.Models.pretrained_model_name.split('/')[-1]}")
    # load only the vit part from the model
    model.model.vit._load_from_state_dict(state_dict=old_model['state_dict'],prefix='model.vit.',strict=True,missing_keys=[],unexpected_keys=[],error_msgs=[],local_metadata={})


#################### Training ###############################
wandb_logger.watch(model=model, log='all')
trainer = pl.Trainer(accelerator='gpu', max_epochs=global_config.Training.epochs,logger=wandb_logger, callbacks=[wandb_checkpoints])

# load pretrained from some layer of the model
trainer_ckpt = global_config.Training.path_loading if global_config.Training.load_existing_path  else None
if global_config.Training.load_existing_path:
    model.load_state_dict(load(global_config.Training.path_loading)["state_dict"])
# if the training has a previous checkpoint (differs from loading only model weights - this is also training params and epoch)
trainer.fit(model=model, train_dataloaders=compatibility_dl,
            val_dataloaders=compatibility_dl,ckpt_path=trainer_ckpt)
