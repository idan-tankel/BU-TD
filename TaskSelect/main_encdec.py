# coding: utf-8
import os
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from transformers import ViTConfig

from datasets import EmnistLeftRight
from modules import LeftRightEncDec
from utils import set_random_seed, train

os.environ["WANDB_MODE"] = "disabled"
torch.autograd.set_detect_anomaly(True)

run_config = {
    'lr': 3e-4,
    'encoder_hidden_size': 240,
    'encoder_intermediate_size': 768,
    'num_encoder_layers': 3,
    'decoder_hidden_size': 120,
    'decoder_intermediate_size': 384,
    'num_decoder_layers': 4,
    'seed': 57,
    'batch_size': 256,
    'dataset_size': 100000,
    'use_butd': False
}

wandb.init(project="RightLeftViT", entity="avagr", config=run_config)

set_random_seed(wandb.config['seed'])

NUM_CLASSES = 47

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
train_data = EmnistLeftRight("/home/agroskin/ViT/6_extended50k1/train/", NUM_CLASSES, transform,
                             wandb.config['dataset_size'])
val_data = EmnistLeftRight("/home/agroskin/ViT/6_extended50k1/val/", NUM_CLASSES, transform)
test_data = EmnistLeftRight("/home/agroskin/ViT/6_extended50k1/test/", NUM_CLASSES, transform)

BATCH_SIZE = wandb.config['batch_size']

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

model = LeftRightEncDec(num_classes=NUM_CLASSES, enc_config=ViTConfig(
    hidden_size=wandb.config['encoder_hidden_size'],
    num_hidden_layers=wandb.config['num_encoder_layers'],
    intermediate_size=wandb.config['encoder_intermediate_size']), dec_config=ViTConfig(
    hidden_size=wandb.config['decoder_hidden_size'],
    num_hidden_layers=wandb.config['num_decoder_layers'],
    intermediate_size=wandb.config['decoder_intermediate_size']), use_butd=wandb.config['use_butd'])

optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config['lr'])
loss = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, verbose=True)

wandb.watch(model)

MODEL_NAME = "leftright_encoder-decoder_last_state"

train_loss, val_loss, test_loss = train(model, train_loader, val_loader, test_loader, loss, optimizer, "cuda:0",
                                        n_epochs=150, scheduler=None, verbose=True,
                                        check_dir=None, save_every=5,
                                        model_name=MODEL_NAME, show_tqdm=False)

with open(f"loss_curves_{MODEL_NAME}.txt", 'w') as f:
    f.write(str(train_loss) + '\n')
    f.write(str(val_loss) + '\n')
    f.write(str(test_loss) + '\n')
