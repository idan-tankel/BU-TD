from transformers import Blip2Config, BlipConfig, BlipForConditionalGeneration, BlipForImageTextRetrieval, AutoProcessor
from pathlib import Path
import git
from swig.JSL.verb.imsituDatasetGood import imSituDatasetGood
from swig.global_utils.mapping import get_mapping
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import ViTForImageClassification, ViTConfig, TrainingArguments
from torch.utils.data import DataLoader
from src.models.create_model import ModelWrapper
from src.models.BLIP_model import BLIPWrapper
from src.Configs.Config import Config
import os
import sys
sys.path.append(os.path.abspath(".."))

# configure directorites relative to the git root
git_repo = git.Repo(__file__, search_parent_directories=True)
git_root = Path(git_repo.working_dir)
results_dir = rf"{git_root.parent}/data/emnist/results"
os.makedirs(results_dir, exist_ok=True)
data_dir = Path(rf"{git_root.parent}/data")
os.makedirs(data_dir, exist_ok=True)
checkpoints_dir = rf"{git_root.parent}/data/checkpoints"
os.makedirs(results_dir, exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument("--title", type=str, default=None)
parser.add_argument("--workers", type=int, default=16)
parser.add_argument("--lr", type=float, default=0.001)

parser.add_argument("--detach_epoch", type=int, default=12)
parser.add_argument("--gt_noun_epoch", type=int, default=5)
parser.add_argument("--hidden-size", type=int, default=1024)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--verb-path", type=str, default=None)
parser.add_argument("--jsl-path", type=str, default=None)
parser.add_argument("--image-file", type=str,
                    default='swig/SWiG_jsons/test.json')
parser.add_argument("--store-features", action="store_true", default=False)
parser.add_argument('-f', '--file')
args = parser.parse_args()

# if args.verb_path == None:
#     print('please input a path to the verb model weights')
#     return
# if args.jsl_path == None:
#     print('please input a path to the jsl model weights')
#     return
# if args.image_file == None:
#     print('please input a path to the image file')
#     return

# if args.store_features:
#     if not os.path.exists('local_features'):
#         os.makedirs('local_features')

kwargs = {"num_workers": args.workers} if torch.cuda.is_available() else {}
verbs = 'swig/global_utils/verb_indices.txt'
verb_to_idx, idx_to_verb = get_mapping(verbs)
try:
    file = args.file
except AttributeError:
    file = r"small_vit.yaml"

if args.file is None:
    file = r"small_vit.yaml"


configuration = Config(experiment_filename=file)
huggingface_config = BlipConfig.from_pretrained(configuration.Models.pretrained_model_name)
# update huggingface_config.text_config.vocab_size
huggingface_config = configuration.merge2huggingface(huggingface_config=huggingface_config)
# instead of merge2huggingface, you may add kwargs to the from_pretrained method
# edit the huggingface config with the current configuration

# load the BLIP model

model = BLIPWrapper(config=configuration, huggingface_config=huggingface_config)
# BLIP preprocessor
preprocessor = AutoProcessor.from_pretrained(configuration.Models.pretrained_model_name)

# wrapper for huggingface - in order to the dataloader would not have to deal with the extra keys
# while loading


# Data - define the dataset after the model since the preprocessing is model dependent
def transform(x): return preprocessor(x)["pixel_values"][0]


test_dataset = imSituDatasetGood(verb_to_idx, json_file=args.image_file,
                                 image_store_location=r"swig/images_512", inference=False, is_train=True, transformation=transform,
                                 preprocessor=preprocessor)
swig_dataloader = DataLoader(
    test_dataset, batch_size=args.batch_size, **kwargs)
print(test_dataset[0])
iterator = iter(swig_dataloader)
print("initializing model")

# get the literal labels in order to pass them to blip

# preprocess the text of the lateral labels
# preprocessor(example["label_all_revert"], truncation=True, padding=True)
# the preprocessor should be before normalization and stuff
# the preprocessing should be within the model forward pass or within the dataset itself
# Logging
# wandb integration
wandb_logger = WandbLogger(project="BU_TD",
                           job_type='train', log_model=True, save_dir=results_dir)
wandb_checkpoints = ModelCheckpoint(
    dirpath=checkpoints_dir, monitor="train_loss_epoch", mode="min")

# Training
trainer = pl.Trainer(accelerator='gpu', max_epochs=configuration.Training.epochs,
                     logger=wandb_logger, callbacks=[wandb_checkpoints])


# load pretrained from some layer of the model
trainer_ckpt = configuration.Training.path_loading if configuration.Training.load_existing_path else None
if configuration.Training.load_existing_path:
    model.load_state_dict(torch.load(configuration.Training.path_loading)["state_dict"])

# keep training - overfitting
trainer.fit(model=model, train_dataloaders=swig_dataloader, val_dataloaders=swig_dataloader, ckpt_path=None)

# define the evaluation loop
trainer.test(model, swig_dataloader)
