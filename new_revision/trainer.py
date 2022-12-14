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
import os
git_repo = git.Repo(os.getcwd(), search_parent_directories=True)
git_root = Path(git_repo.working_dir)
results_dir = rf"{git_root.parent}/data/emnist/results"
os.makedirs(results_dir, exist_ok=True)
data_dir = Path(rf"{git_root.parent}/data")
os.makedirs(data_dir, exist_ok=True)
checkpoints_dir = rf"{git_root.parent}/data/emnist/checkpoints"
os.makedirs(results_dir, exist_ok=True)

global_config = Config()

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize((224, 224))])
# train_ds = datasets.Food101(
#     download=True, root=data_dir, split="train", transform=transform)
# val_ds = datasets.Food101(
#     download=True, root=data_dir, split="test", transform=transform)
# test_ds = datasets.EMNIST(download=True, root=data_dir,
#                           split='balanced', train=False, transform=transform)

# train_dl = DataLoader(train_ds, batch_size=global_config.Training.batch_size,
#                       shuffle=True, num_workers=global_config.Training.num_workers)
# test_dl = DataLoader(test_ds, batch_size=global_config.Training.batch_size,
#                      shuffle=False, num_workers=global_config.Training.num_workers)
# test_dl = DataLoader(val_ds, batch_size=global_config.Training.batch_size,
#                      shuffle=False, num_workers=global_config.Training.num_workers)

compatibility_dataset = DatasetAllDataSetTypesAll(root=rf'/home/idanta/data/6_extended_testing/train/', opts=global_config,  direction=1,
                                                  is_train=True, obj_per_row=6, obj_per_col=1, split=False,nexamples=100000)
compatibility_dl = DataLoader(compatibility_dataset, batch_size=global_config.Training.batch_size,
                              num_workers=global_config.Training.num_workers)
wandb_logger = WandbLogger(project="BU_TD",
                           job_type='train', log_model=True, save_dir=results_dir)
wandb_checkpoints = ModelCheckpoint(
    dirpath=checkpoints_dir, monitor="train_loss_epoch", mode="min")


if global_config.Models.pretrained_model_name is not None:
    model = ViTForImageClassification.from_pretrained(
        global_config.Models.pretrained_model_name)
# else create the model according to the config params
else:
    model = ViTForImageClassification(
        ViTConfig(
            image_size=global_config.Models.image_size,
            hidden_size=768,
            num_hidden_layers=4,
            num_classes=global_config.Models.num_classes,
            qkv_bias=True
        )
    )
model = ModelWrapper(model=model, config=global_config)



wandb_logger.watch(model=model, log='all')
trainer = pl.Trainer(accelerator='gpu', max_epochs=global_config.Training.epochs,
                     logger=wandb_logger, callbacks=[wandb_checkpoints])

# load pretrained from some layer of the model
old_model = load(f"{checkpoints_dir}/epoch=176-step=276651.ckpt")
model.model.vit._load_from_state_dict(state_dict=old_model['state_dict'],prefix='model.vit.',strict=True,missing_keys=[],unexpected_keys=[],error_msgs=[],local_metadata={})
trainer_ckpt = (if global_config.Training.load_existing_path global_config.Training.path_loading else None)

trainer.fit(model=model, train_dataloaders=compatibility_dl,
            val_dataloaders=compatibility_dl,ckpt_path=trainer_ckpt)
