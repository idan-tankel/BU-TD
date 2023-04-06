"""
Here we define the function utils.
"""
import argparse
from typing import Iterator, Optional

import numpy as np

import torch.optim as optim

from .data.Enums import data_set_types, Model_type, TrainingFlag
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import torch.utils.data as data

from .data.dataset import *


def num_params(params: Iterator) -> int:
    """
    Computing the number of parameters in a given list.
    Args:
        params: The list of parameters.

    Returns: The number of learnable parameters in the list.

    """
    num_param = 0
    for param in params:
        # For each parameter in the model we multiply all its shape dimensions.
        shape = torch.tensor(param.shape)  # Make a tensor.
        num_param += torch.prod(shape)  # Add to the sum.
    return num_param


def create_optimizer_and_scheduler(opts: argparse, learned_params: list, nbatches: int) -> \
        [torch.optim, torch.optim.lr_scheduler]:
    """
    Create optimizer and a scheduler according to opts.
    Args:
        opts: The model options.
        learned_params: The learned parameters.
        nbatches: The number of batches in the data-loader

    Returns: Optimizer, scheduler.

    """
    optimizer = None
    if opts.data_set_obj['optimizer'] == 'SGD':
        optimizer = optim.SGD(params=learned_params, lr=opts.data_set_obj['initial_lr'], momentum=0.9,
                              weight_decay=opts.data_set_obj['wd'])

    if opts.data_set_obj['optimizer'] == 'Adam':
        optimizer = optim.Adam(params=learned_params, lr=opts.data_set_obj['initial_lr'],
                               weight_decay=opts.data_set_obj['wd'])

    scheduler = None
    if opts.scheduler_type is optim.lr_scheduler.StepLR:
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=nbatches * 20,
                                              gamma=opts.data_set_obj.factor)
    if opts.scheduler_type is optim.lr_scheduler.CosineAnnealingLR:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    if opts.scheduler_type is optim.lr_scheduler.MultiStepLR:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opts.data_set_obj['milestones'],
                                                   last_epoch=-1, gamma=opts.data_set_obj['factor'])

    return optimizer, scheduler


def load_model(model: nn.Module, results_dir: str, model_path: str):
    """
    Loads and returns the model checkpoint as a dictionary.
    Args:
        model: The model we want to load into.
        model_path: The path to the model.
        results_dir: Trained model dir.

    Returns: The loaded checkpoint.

    """
    model_path = os.path.join(results_dir, model_path)  # The path to the model.
    checkpoint = torch.load(model_path)  # Loading the saved data.
    model.load_state_dict(checkpoint)


def Expand(opts: argparse, mod: Tensor, shapes: list, shape: list) -> Tensor:
    """
    Expand the tensor in interleaved manner to match the neuron's shape.
    Args:
        opts: The model opts.
        mod: The modulations.
        shapes: The shape to multiply each dimension.
        shape: The desired.

    Returns: The expanded modulations.

    """
    expand = False
    if expand:
        for dim, shape in enumerate(shapes):
            mod = torch.repeat_interleave(mod, shape, dim=dim)
    else:
        e1, e2, e3, e4 = mod.shape
        m1, m2, m3, m4 = shape
        mod = torch.nn.functional.interpolate(mod.view((e4, e3, e2, e1)), mode=opts.data_set_obj['interpolation'],
                                              size=(m1, m2))
    e1, e2, e3, e4 = mod.shape
    mod = mod.view((e3, e4, e2, e1))
    return mod


def from_from_data_set_type_to_object(ds_type: data_set_types):
    """
    From data-set type to class object.
    Args:
        ds_type: The data-set type.

    Returns: dataset object.

    """
    if ds_type is data_set_types.StanfordCars:
        return StanfordCarsDataSet
    if ds_type is data_set_types.Flowers:
        return FlowersDataSet
    if ds_type is data_set_types.CUB200:
        return CUB200Dataset
    if ds_type is data_set_types.Food101:
        return Food101Dataset
    if ds_type is data_set_types.WikiArt or ds_type is data_set_types.Sketches or ds_type is data_set_types.ImageNet:
        return ImageFolderDataSets
    if ds_type is data_set_types.SUN:
        return SUN


def Get_dataloaders(opts: argparse, task_id: int):
    """
    Get the data-set object.
    Args:
        opts: The model opts.
        task_id: The task id.

    Returns: train, test data-sets.

    """
    ds_type = opts.ds_type
    project_path = opts.project_path
    ntasks = opts.data_set_obj['ntasks']
    dataset_obj = from_from_data_set_type_to_object(ds_type=opts.ds_type)
    train_ds = dataset_obj(root=os.path.join(project_path, f'data/RAW/{str(ds_type)}'),
                           is_train=True,
                           task=task_id, ntasks=ntasks)
    test_ds = dataset_obj(root=os.path.join(project_path, f'data/RAW/{str(ds_type)}'),
                          is_train=False,
                          task=task_id, ntasks=ntasks)
    if ds_type in [data_set_types.CIFAR10, data_set_types.CIFAR100, data_set_types.DTD]:
        seed = torch.Generator().manual_seed(0)
        nsamples_train, nsamples_val = int(np.rint(len(train_ds) * 0.9)), int(np.rint(len(train_ds) * 0.1))
        train_ds, val_ds = data.random_split(train_ds, [nsamples_train, nsamples_val], generator=seed)
        train_dl = DataLoader(dataset=train_ds, batch_size=opts.data_set_obj.bs, shuffle=True, pin_memory=True)
        val_dl = DataLoader(dataset=val_ds, batch_size=opts.data_set_obj.bs * 2, shuffle=False)
        test_dl = DataLoader(dataset=test_ds, batch_size=opts.data_set_obj.bs * 2, shuffle=False)
    else:
        train_dl = DataLoader(dataset=train_ds, batch_size=opts.data_set_obj['bs'], shuffle=True, pin_memory=True,
                              num_workers=2)
        val_dl = DataLoader(dataset=test_ds, batch_size=opts.data_set_obj['bs'] * 2, shuffle=False, pin_memory=True,
                            num_workers=2)
        test_dl = DataLoader(dataset=test_ds, batch_size=opts.data_set_obj['bs'] * 2, shuffle=False, pin_memory=True,
                             num_workers=2)

    return train_dl, val_dl, test_dl


def Get_Model_path(opts: argparse, ds_type: data_set_types, training_flag: TrainingFlag,
                   model_type: Model_type, task_id: int) -> Tuple[str, str]:
    """
    Getting the path we store the model into.
    Args:
        opts: The model opts.
        ds_type: The data-set type.
        training_flag: The training flag.
        model_type: The model type.
        task_id: The task id.

    Returns: The path to store the model into.

    """
    opti_type = opts.data_set_obj['optimizer']
    ntasks = opts.data_set_obj['ntasks']
    threshold = opts.data_set_obj['threshold']
    factor = opts.data_set_obj['factor']
    wd = opts.data_set_obj['wd']
    bs = opts.data_set_obj['bs']
    milestones = opts.data_set_obj['milestones']
    initial_lr = opts.data_set_obj['initial_lr']
    drop_out_rate = opts.data_set_obj['drop_out_rate']
    mask_weight_modulation = opts.data_set_obj['mask_modulation_factor']
    weight_modulation = opts.data_set_obj['weight_modulation_factor']
    name1 = f'{str(ds_type)}/{str(training_flag)}/{str(model_type)}/' \
            f'{opti_type}/Num_tasks_{ntasks}/Task{task_id}/'
    name = name1 + f'threshold_{threshold}_factor_{factor}_wd' \
                   f'_{wd}_bs_' \
                   f'{bs}_lr=' \
                   f'_{initial_lr}_milestones_' \
                   f'{milestones}_drop_out_rate_{drop_out_rate}_modulation_factor_' \
                   f'{weight_modulation}_mask_modulation_factor_{mask_weight_modulation}'

    return name1, name


def Get_Learned_Params(model: nn.Module, training_flag: TrainingFlag, task_id: int) -> List[nn.Parameter]:
    """
    Get the learned parameters.
    Args:
        model: The model.
        training_flag: The training flag.
        task_id: The task id.

    Returns: list of the learned parameters

    """
    learned_params = []
    if training_flag is TrainingFlag.Full_Model:
        learned_params.extend(list(model.parameters()))
    if training_flag is TrainingFlag.Modulation or training_flag is TrainingFlag.MaskAndMod:
        learned_params.extend(model.modulations[task_id])
    if training_flag is TrainingFlag.Classifier_Only:
        learned_params.extend(model.linear.parameters())
    if training_flag is TrainingFlag.Masks or training_flag is TrainingFlag.MaskAndMod:
        learned_params.extend(model.masks[task_id])
    return learned_params


def Define_Trainer(opts: argparse, name: str) -> pl.Trainer:
    """
    Define the trainer.
    Args:
        opts: The model opts.
        name: The model name.

    Returns: pytorch lightning trainer.

    """
    wandbLogger = WandbLogger(project="Affecting conv weight", job_type='train', name=name,
                              save_dir=os.path.join(opts.project_path, 'data/loggers'))
    checkpoint_second = pl.callbacks.ModelCheckpoint(dirpath=os.path.join(opts.results_dir, name),
                                                     mode='max', monitor='val_acc',
                                                     filename='{val_acc:.3f}')

    trainer = pl.Trainer(max_epochs=opts.data_set_obj['epochs'], accelerator='gpu',
                         logger=wandbLogger, callbacks=[checkpoint_second], precision=16)
    return trainer


def Load_pretrained_resnet(opts: argparse, model: nn.Module, ds_type: data_set_types,
                           model_type: Optional[Model_type]) -> None:
    """
    Load pretrained model.
    Args:
        opts: The model opts.
        model: The model.
        ds_type: The data-set type.
        model_type: The model type.

    """
    checkpoint = {}
    state_dict = {}
    prefix = ''
    num_blocks = [5, 5, 5]
    if ds_type is not data_set_types.CIFAR100:
        weights = model_type.weights().IMAGENET1K_V1
        state_dict = weights.get_state_dict(progress=True)
        num_blocks = model_type.num_blocks()
    if ds_type is data_set_types.CIFAR100:
        prefix = 'module.'
        model_path = os.path.join(opts.results_dir, 'resnet32.th')
        state_dict = torch.load(model_path)['state_dict']  # Loading the saved data.
    for layer_id in range(len(num_blocks)):
        for block_id in range(num_blocks[layer_id]):
            for layer_type in ['conv1', 'conv2', 'conv3']:
                for param_type in ['weight']:
                    new_key = f'layers.{layer_id}.{block_id}.{layer_type}.{param_type}'
                    old_key = f'{prefix}layer{layer_id + 1}.{block_id}.{layer_type}.{param_type}'
                    try:
                        if new_key in model.state_dict().keys():
                            checkpoint[new_key] = state_dict[old_key]
                    except KeyError:
                        pass
            for layer_type in ['bn1', 'bn2', 'bn3']:
                for param_type in ['weight', 'bias']:
                    new_key = f'layers.{layer_id}.{block_id}.{layer_type}.norm.{param_type}'
                    old_key = f'{prefix}layer{layer_id + 1}.{block_id}.{layer_type}.{param_type}'
                    try:
                        if new_key in model.state_dict().keys():
                            checkpoint[new_key] = state_dict[old_key]
                    except KeyError:
                        pass

            for layer_type in ['downsample.conv1x1']:
                for param_type in ['weight']:
                    new_key = f'layers.{layer_id}.{block_id}.{layer_type}.{param_type}'
                    old_key = f'{prefix}layer{layer_id + 1}.{block_id}.downsample.0.{param_type}'
                    if new_key in model.state_dict().keys():
                        checkpoint[new_key] = state_dict[old_key]

            for layer_type in ['downsample.norm.norm']:
                for param_type in ['weight', 'bias']:
                    new_key = f'layers.{layer_id}.{block_id}.{layer_type}.{param_type}'
                    old_key = f'{prefix}layer{layer_id + 1}.{block_id}.downsample.1.{param_type}'
                    if new_key in model.state_dict().keys():
                        checkpoint[new_key] = state_dict[old_key]

            checkpoint['conv1.weight'] = state_dict[f'{prefix}conv1.weight']
            checkpoint['bn1.norm.weight'] = state_dict[f'{prefix}bn1.weight']
            checkpoint['bn1.norm.bias'] = state_dict[f'{prefix}bn1.bias']

        for key, val in model.state_dict().items():
            if "running_mean" in key or 'running_var' in key or 'linear' in key or 'modulations' in key or \
                    'modulations' in key:
                checkpoint[key] = val

    model.load_state_dict(checkpoint)  # Loading the saved weights.


def Load_Pretrained_EfficientNet(model: nn.Module, model_type: Model_type) -> None:
    """
    Load EfficientNet.
    Args:
        model: The model.
        model_type: The model type.

    Returns: None

    """
    num_layers = 7
    weights = model_type.weights().IMAGENET1K_V1
    old_state_dict = weights.get_state_dict(progress=True)
    current_state_dict = {}
    # current_state_dict = copy.deepcopy(old_state_dict)
    for name, param in model.state_dict().items():
        if "mask" in name or "running" in name or "classifier" in name:
            current_state_dict[name] = param

    for layer_id in range(num_layers):
        num_blocks = model.inverted_residual_setting[layer_id].num_layers
        # print(num_blocks)

        for block_id in range(num_blocks):
            # Begin with conv.
            for bit in [0, 2]:
                try:
                    old_bit = 0 if bit == 0 else 3
                    old_key = f"features.{layer_id + 1}.{block_id}.block.{old_bit}.0" \
                              f".weight"
                    new_key = f'features.{layer_id + 1}.{block_id}.block.{old_bit}.conv.weight'
                    current_state_dict[new_key] = old_state_dict[old_key]
                except:
                    pass
            #
            try:
                new_key = f'features.{layer_id + 1}.{block_id}.block.1.conv.weight'
                old_key = f'features.{layer_id + 1}.{block_id}.block.1.0.weight'
                current_state_dict[new_key] = old_state_dict[old_key]
            except KeyError:
                pass
            try:
                new_key = f'features.{layer_id + 1}.{block_id}.block.1.bn.weight'
                old_key = f'features.{layer_id + 1}.{block_id}.block.1.1.weight'
                current_state_dict[new_key] = old_state_dict[old_key]
            except KeyError:
                pass
            try:
                new_key = f'features.{layer_id + 1}.{block_id}.block.1.bn.bias'
                old_key = f'features.{layer_id + 1}.{block_id}.block.1.1.bias'
                current_state_dict[new_key] = old_state_dict[old_key]
            except KeyError:
                pass
            try:
                new_key = f'features.{layer_id + 1}.{block_id}.block.2.bn.bias'
                old_key = f'features.{layer_id + 1}.{block_id}.block.2.1.bias'
                current_state_dict[new_key] = old_state_dict[old_key]
            except KeyError:
                pass
            try:
                new_key = f'features.{layer_id + 1}.{block_id}.block.2.bn.weight'
                old_key = f'features.{layer_id + 1}.{block_id}.block.2.1.weight'
                current_state_dict[new_key] = old_state_dict[old_key]
            except KeyError:
                pass
            #
            try:
                new_key = f'features.{layer_id + 1}.{block_id}.block.3.bn.bias'
                old_key = f'features.{layer_id + 1}.{block_id}.block.3.1.bias'
                current_state_dict[new_key] = old_state_dict[old_key]
            except KeyError:
                pass
            try:
                new_key = f'features.{layer_id + 1}.{block_id}.block.3.bn.weight'
                old_key = f'features.{layer_id + 1}.{block_id}.block.3.1.weight'
                current_state_dict[new_key] = old_state_dict[old_key]
            except KeyError:
                pass
            #
            # Continue to BN layers.
            old_key = f"features.{layer_id + 1}.{block_id}.block.0.1.weight"
            new_key = f'features.{layer_id + 1}.{block_id}.block.0.bn.weight'
            current_state_dict[new_key] = old_state_dict[old_key]
            old_key = f"features.{layer_id + 1}.{block_id}.block.0.1.bias"
            new_key = f'features.{layer_id + 1}.{block_id}.block.0.bn.bias'
            current_state_dict[new_key] = old_state_dict[old_key]
            # FC layer
            for bit in [1, 2]:
                for layer in ['weight', 'bias']:
                    old_key = f'features.{layer_id + 1}.{block_id}.block.2.fc{bit}.{layer}'
                    new_key = f'features.{layer_id + 1}.{block_id}.block.2.fc{bit}.{layer}'
                    try:
                        current_state_dict[new_key] = old_state_dict[old_key]
                    except KeyError:
                        pass

    current_state_dict['features.8.conv.weight'] = old_state_dict['features.8.0.weight']
    current_state_dict['features.8.bn.weight'] = old_state_dict['features.8.1.weight']
    current_state_dict['features.8.bn.bias'] = old_state_dict['features.8.1.bias']
    current_state_dict['features.0.conv.weight'] = old_state_dict['features.0.0.weight']
    current_state_dict['features.0.bn.weight'] = old_state_dict['features.0.1.weight']
    current_state_dict['features.0.bn.bias'] = old_state_dict['features.0.1.bias']
    current_state_dict['features.1.0.block.2.conv.weight'] = old_state_dict['features.1.0.block.2.0.weight']
    current_state_dict['features.1.0.block.1.fc1.weight'] = old_state_dict['features.1.0.block.1.fc1.weight']
    current_state_dict['features.1.0.block.1.fc1.bias'] = old_state_dict['features.1.0.block.1.fc1.bias']
    current_state_dict['features.1.0.block.1.fc2.weight'] = old_state_dict['features.1.0.block.1.fc2.weight']
    current_state_dict['features.1.0.block.1.fc2.bias'] = old_state_dict['features.1.0.block.1.fc2.bias']
    model.load_state_dict(current_state_dict)


def Load_Pretrained_MobileNet(model, model_type: Model_type) -> None:
    """
    Load MobileNet.
    Args:
        model: The model.
        model_type: The model type.

    Returns: None

    """
    weights = model_type.weights().IMAGENET1K_V1
    old_weights = weights.get_state_dict(progress=True)
    new_check = weights.get_state_dict(progress=True)
    for name, param in model.state_dict().items():
        if name not in old_weights.keys():
            new_check[name] = param
    model.load_state_dict(new_check)  # Loading the saved weights.


def Load_Pretrained_model(opts: argparse, model: nn.Module, ds_type: data_set_types,
                          model_type: Optional[Model_type]) -> None:
    """

    Args:
        opts: The opts.
        model: The model.
        ds_type: The data-set type.
        model_type: The model type.

    Returns:

    """
    if model_type is Model_type.MobileNet:
        Load_Pretrained_MobileNet(model=model, model_type=model_type)
    if model_type is Model_type.EfficientNet:
        Load_Pretrained_EfficientNet(model, model_type)
    if model_type in [Model_type.ResNet14, Model_type.ResNet18, Model_type.ResNet20, Model_type.ResNet32,
                      Model_type.ResNet34, Model_type.ResNet50, Model_type.ResNet101]:
        Load_pretrained_resnet(opts=opts, model=model, model_type=model_type, ds_type=ds_type)
