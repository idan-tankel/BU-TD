# coding: utf-8
import os
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from transformers import ViTConfig

from datasets import EmnistLeftRight
from modules import LeftRightBUTD
from utils import set_random_seed, train

# os.environ["WANDB_MODE"] = "disabled"
# torch.autograd.set_detect_anomaly(True)

run_config = {
    'lr': 3e-4,
    'hidden_size': 192,
    'intermediate_size': 512,
    'num_layers': 3,
    'seed': 57,
    'batch_size': 256,
    'dataset_size': 100000,
    'mix_layer': 'v',
    'add_self_attention': True,
    'type': 'multiplication'
}

wandb.init(project="RightLeftViT", entity="avagr", config=run_config)

set_random_seed(run_config['seed'])

NUM_CLASSES = 47

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
train_data = EmnistLeftRight("/home/agroskin/ViT/6_extended50k1/train/", NUM_CLASSES, transform,
                             run_config['dataset_size'])
val_data = EmnistLeftRight("/home/agroskin/ViT/6_extended50k1/val/", NUM_CLASSES, transform)
test_data = EmnistLeftRight("/home/agroskin/ViT/6_extended50k1/test/", NUM_CLASSES, transform)

BATCH_SIZE = run_config['batch_size']

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

model = LeftRightBUTD(num_classes=NUM_CLASSES, config=ViTConfig(
    hidden_size=run_config['hidden_size'],
    num_hidden_layers=run_config['num_layers'],
    intermediate_size=run_config['intermediate_size']),
                      mix_with=run_config['mix_layer'],
                      use_self_attention=run_config['add_self_attention'],
                      total_token_size=run_config['hidden_size'] * 197)

optimizer = torch.optim.Adam(model.parameters(), lr=run_config['lr'])
loss = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, verbose=True)

wandb.watch(model)

MODEL_NAME = "leftright_butd"

train_loss, val_loss, test_loss = train(model, train_loader, val_loader, test_loader, loss, optimizer, "cuda:0",
                                        n_epochs=150, scheduler=None, verbose=True,
                                        check_dir=None, save_every=5,
                                        model_name=MODEL_NAME, show_tqdm=False)

with open(f"loss_curves_{MODEL_NAME}.txt", 'w') as f:
    f.write(str(train_loss) + '\n')
    f.write(str(val_loss) + '\n')
    f.write(str(test_loss) + '\n')
