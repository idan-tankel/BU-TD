import os
import sys
sys.path.append(os.path.abspath(".."))
from src.Configs.Config import Config
from src.models.create_model import ModelWrapper
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTConfig, TrainingArguments
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import torch
from transformers import Blip2Config,BlipConfig,BlipForConditionalGeneration,BlipForImageTextRetrieval,AutoProcessor
import argparse
from swig.JSL.verb.imsituDatasetGood import imSituDatasetGood
import git
from pathlib import Path


git_repo = git.Repo(__file__, search_parent_directories=True)
git_root = Path(git_repo.working_dir)
results_dir = rf"{git_root.parent}/data/emnist/results"
os.makedirs(results_dir, exist_ok=True)
data_dir = Path(rf"{git_root.parent}/data")
os.makedirs(data_dir, exist_ok=True)
checkpoints_dir = rf"{git_root.parent}/data/checkpoints"
os.makedirs(results_dir, exist_ok=True)


def get_mapping(word_file):
    """
    get_mapping is a function that returns a dictionary mapping words to indices and a list of words

    Args:
        word_file (_type_): _description_

    Returns:
        _type_: _description_
    """    
    dict = {}
    word_list = []
    with open(word_file) as f:
        k = 0
        for line in f:
            word = line.split('\n')[0]
            dict[word] = k
            word_list.append(word)
            k += 1
    return dict, word_list


parser = argparse.ArgumentParser()
parser.add_argument("--title", type=str, default=None)
parser.add_argument("--workers", type=int, default=16)
parser.add_argument("--lr", type=float, default=0.001)

parser.add_argument("--detach_epoch", type=int, default=12)
parser.add_argument("--gt_noun_epoch", type=int, default=5)
parser.add_argument("--hidden-size", type=int, default=1024)
parser.add_argument("--batch-size", type=int, default=16)
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

# load the BLIP model

if configuration.Models.pretrained_model_name is not None and "local" not in str(configuration.Models.pretrained_model_name):
    model = BlipForImageTextRetrieval.from_pretrained(
        configuration.Models.pretrained_model_name)
else:
    model = ViTForImageClassification(
        ViTConfig(
            image_size=configuration.Models.image_size,
            hidden_size=768,
            num_hidden_layers=configuration.Models.depth,
            num_classes=configuration.Models.num_classes,
            qkv_bias=False,
            num_attention_heads=configuration.Models.num_attention_heads
        )
    )
# model = ModelWrapper(model=model, config=configuration)
# BLIP preprocessor
preprocessor = AutoProcessor.from_pretrained(configuration.Models.pretrained_model_name)


################################################### Data

test_dataset = imSituDatasetGood(verb_to_idx, json_file=args.image_file,
                                 image_store_location=rf"swig/images_512", inference=False, is_train=True)
test_dataloader = DataLoader(
    test_dataset, batch_size=args.batch_size, **kwargs)
iterator = iter(test_dataloader)
print("initializing jsl model")
example = next(iterator)
print(example)

# get the literal labels in order to pass them to blip
[idx_to_verb[int(x)] for x in example["label_all"].flatten().tolist()]
# preprocess the text of the lateral labels
# preprocessor(example["label_all_revert"], truncation=True, padding=True)
# the preprocessor should be before normalization and stuff
################################################### Logging
# wandb integration
wandb_logger = WandbLogger(project="BU_TD",
                           job_type='train', log_model=True, save_dir=results_dir)
wandb_checkpoints = ModelCheckpoint(
    dirpath=checkpoints_dir, monitor="train_loss_epoch", mode="min")

################################################### Training
trainer = pl.Trainer(accelerator='gpu', max_epochs=configuration.Training.epochs,
                     logger=wandb_logger, callbacks=[wandb_checkpoints])



# load pretrained from some layer of the model
trainer_ckpt = configuration.Training.path_loading if configuration.Training.load_existing_path  else None
if configuration.Training.load_existing_path:
    model.load_state_dict(torch.load(configuration.Training.path_loading)["state_dict"])

# keep training - overfitting
trainer.fit(model=model, train_dataloaders=test_dataloader,val_dataloaders=test_dataloader,ckpt_path=None)

#### define the evaluation loop
trainer.test(model, test_dataloader)



