import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path
from transformers import ViTForImageClassification, ViTConfig, TrainingArguments
from torchvision import datasets, transforms
from models.create_model import ModelWrapper
from torch.utils.data import DataLoader
from torch import cuda, device
from types import SimpleNamespace
import git
import os
git_repo = git.Repo(os.getcwd(), search_parent_directories=True)
git_root = Path(git_repo.working_dir)
results_dir = rf"{git_root.parent}/data/emnist/results/"
os.makedirs(results_dir, exist_ok=True)
data_dir = Path(rf"{git_root.parent}/data")
os.makedirs(data_dir, exist_ok=True)
checkpoints_dir = rf"{git_root.parent}/data/emnist/checkpoints/"
os.makedirs(results_dir, exist_ok=True)

global_config = {
    "batch_size": 10,
    "learning_rate": 1e-4,
    "num_workers": 10,
    "num_epochs": 60,
    "image_size": 224,
    "num_channels": 1,
    "num_gpus": 1,
    "num_nodes": 1,
    "precision": 16,
    "optimizer": "adam",
    "scheduler": "cosine",
    "scheduler_warmup_steps": 0,
    "scheduler_t_max": 10,
    
}
global_config = SimpleNamespace(**global_config)

transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224))])
train_ds = datasets.Food101(
    download=True, root=data_dir, split="train", transform=transform)
val_ds = datasets.Food101(
    download=True, root=data_dir,split="test", transform=transform)
test_ds = datasets.EMNIST(download=True, root=data_dir,
                          split='balanced', train=False, transform=transform)

train_dl = DataLoader(train_ds, batch_size=global_config.batch_size,
                      shuffle=True, num_workers=global_config.num_workers)
test_dl = DataLoader(test_ds, batch_size=global_config.batch_size,
                     shuffle=False, num_workers=global_config.num_workers)
test_dl = DataLoader(val_ds, batch_size=global_config.batch_size,
                     shuffle=False, num_workers=global_config.num_workers)


wandb_logger = WandbLogger(project="My_first_project_5.10",
                           job_type='train', log_model=True, save_dir=results_dir)
wandb_checkpoints = ModelCheckpoint(
    dirpath=checkpoints_dir, monitor="train_loss_epoch", mode="min")

model = ModelWrapper(model=ViTForImageClassification.from_pretrained('google/vit-base-patch16-224'))




wandb_logger.watch(model=model, log='all')
trainer = pl.Trainer(accelerator='gpu', max_epochs=60,
                     logger=wandb_logger, callbacks=[wandb_checkpoints])

trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=test_dl)
