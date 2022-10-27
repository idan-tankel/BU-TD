import argparse
import os
import pickle
import sys
from types import SimpleNamespace
import yaml
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import git
from Configs.Config import Config
from typing import Union
from supp.Dataset_and_model_type_specification import AllOptions

sys.path.append(
    r'/home/sverkip/data/BU-TD/yonathan/Recognicion/code/create_dataset')


class DataSetBase(Dataset):

    """
    Base class OmniglotDataSetBase.
    Supports initialization and get item methods.
    """

    def __init__(self, root: str, nclasses_existence: int, ndirections: int, is_train: bool, nexamples: int = None,
                 split: bool = True) -> None:
        """
        Args:
            root: The root to the data.
            nclasses_existence:
            ndirections: The number of classes.
            nexamples: The number of classes.
            split: Whether to split the  dataset.
        """
        self.root = root
        self.nclasses_existence = nclasses_existence
        self.ndirections = ndirections
        self.split = split
        self.splitsize = 1000
        self.is_train = is_train
        if nexamples is None:
            filenames = [os.path.join(dp, f)
                         for dp, dn, fn in os.walk(root) for f in fn]
            images = [f for f in filenames if f.endswith('_img.jpg')]
            self.nexamples = len(images)
        else:
            self.nexamples = nexamples

    def get_root_by_index(self, index: int) -> str:
        """
        Args:
            index: The sample index.

        Returns: Path to the sample.
        """
        if self.split:
            root = os.path.join(self.root, '%d' % (index // self.splitsize))
        else:
            root = self.root
        return root

    def get_raw_sample(self, index):
        """
        Args:
            index: The sample index.

        Returns: The Sample.

        """
        root = self.get_root_by_index(index)
        data_fname = os.path.join(root, '%d_raw.pkl' % index)
        with open(data_fname, "rb") as new_data_file:
            raw_sample = pickle.load(new_data_file)
        return raw_sample

    def __len__(self):
        """
        Returns: Length of the dataset.

        """

        if self.is_train:
            nexamples = self.nexamples
        else:
            nexamples = self.nexamples
        self.targets = [0 for _ in range(nexamples)]
        return nexamples


def struct_to_input(sample: object) -> tuple:
    """
    From sample to inputs.
    Args:
        sample: Sample to return its attributes including

    Returns: The sample's attributes: label_existence, label_all ,flag, label_task.

    """
    label_existence = sample.label_existence
    label_all = sample.label_ordered
    flag = sample.flag
    label_task = sample.label_task
    return label_existence, label_all, flag, label_task


class DatasetAllDataSetTypes(DataSetBase):
    def __init__(self, root: str, opts: Union[argparse.ArgumentParser,Config], arg_and_head_index: int = 0, direction: int = 0, is_train=True, nexamples: int = None, obj_per_row=6,
                 obj_per_col=1, split: bool = True):
        """
        Omniglot data-set.
        Args:
            root: Path to the data.
            opts: The model options.
            arg_and_head_index: The language index.
            direction: The direction id.
            nexamples: The number of examples.
            obj_per_row: Number of objects per row.
            obj_per_col: Number of columns per row.
            split: Whether to split the dataset.
        """

        super(DatasetAllDataSetTypes, self).__init__(
            root, nexamples, split, is_train=is_train)
        if not isinstance(opts,argparse.ArgumentParser):
            git_repo = git.Repo(__file__, search_parent_directories=True)
            git_root = git_repo.working_dir
            full_path = f"{git_root}/yonathan/Recognicion/code/Configs/create_config.yaml"
            with open(full_path, 'r') as stream:
                config_as_dict = yaml.safe_load(stream)
                opts = SimpleNamespace(**config_as_dict)
        self.ntasks = opts.ntasks
        self.nclasses_existence = opts.nclasses
        self.direction = torch.tensor(direction)
        self.ndirections = opts.ndirections
        self.obj_per_row = obj_per_row
        self.obj_per_col = obj_per_col
        self.task_idx = torch.tensor(arg_and_head_index)

    def __getitem__(self, index):
        """
        Args:
            index: sample index.

        Returns: img, label_task, flag, label_all, label_existence.

        """
        # Getting root to the sample
        root = self.get_root_by_index(index)
        fname = os.path.join(root, '%d_img.jpg' % index)
        # Opening the image and converting to Tensor
        img = Image.open(fname).convert('RGB')
        img = T.ToTensor()(img)
        # TODO: all those functions have changed in the beta branch to save all the images as tensors already in the Create dataset phase
        img = 255 * img  # converting to RGB
        # Get raw sample.
        sample = self.get_raw_sample(index)
        # Converting the sample into input.
        (label_existence, label_all, flag, label_task) = struct_to_input(sample)
        char = flag[1].item()  # The referred character
        # Getting the task embedding.
        task_type_ohe = torch.nn.functional.one_hot(self.task_idx, self.ntasks)
        # Getting the direction embedding.
        direction_type_ohe = torch.nn.functional.one_hot(
            self.direction, self.ndirections)
        # Getting the character embedding.
        char_type_one = torch.nn.functional.one_hot(
            torch.tensor(char), self.nclasses_existence)
        # Concatenating into one flag.
        flag = torch.concat(
            [ task_type_ohe, char_type_one], dim=0).float()
        edge_class = self.nclasses_existence
        r, c = (label_all == char).nonzero()
        if self.direction == 0:
            # right
            if c == (self.obj_per_row - 1):
                label_task = edge_class
            else:
                label_task = label_all[r, c + 1]

        if self.direction == 1:
            # left
            if c == 0:
                label_task = edge_class
            else:
                label_task = label_all[r, c - 1]

        elif self.direction == 2:
            # Up
            if r == (self.obj_per_col - 1):
                label_task = edge_class
            else:
                label_task = label_all[r + 1, c]

        elif self.direction == 3:
            # Down
            if r == 0:
                label_task = edge_class
            else:
                label_task = label_all[r - 1, c]
        # TODO change this to use conv2d with the proper filter
        label_existence, label_all, label_task = map(
            torch.tensor, (label_existence, label_all, label_task))
        label_task = label_task.view([-1])
        label_existence = label_existence.float()
        return img, label_task, flag, label_all, label_existence


class DatasetAllDataSetTypesAll(DatasetAllDataSetTypes):
    def calc_label_task_all(self, label_all, not_available_class):
        """
        Compute for each character its neighbor is exists in the sample.
        Args:
            label_all: The label all flag.
            not_available_class: The not available classes.

        Returns: The label all task.

        """
        nclasses_existence = 47
        edge_class = nclasses_existence
        label_adj_all = not_available_class * torch.ones(nclasses_existence)
        for r, row in enumerate(label_all):
            for c, char in enumerate(row):
                if self.direction == 0:
                    # right
                    if c == (self.obj_per_row - 1):
                        res = edge_class
                    else:
                        res = label_all[r, c + 1]
                elif self.direction == 1:
                    # left
                    if c == 0:
                        res = edge_class
                    else:
                        res = label_all[r, c - 1]

                elif self.direction == 2:
                    # Up

                    if r == (self.obj_per_col - 1):
                        res = edge_class
                    else:
                        res = label_all[r + 1, c]

                elif self.direction == 3:
                    # Down
                    if r == 0:
                        res = edge_class
                    else:
                        res = label_all[r - 1, c]

                label_adj_all[char] = res
        return label_adj_all

    def __getitem__(self, index):
        img, label_task, flag, label_all, label_existence = super().__getitem__(index)
        not_available_class = 47
        label_task = self.calc_label_task_all(
            label_all, not_available_class).long()
        return img, label_task, flag, label_all, label_existence
