import argparse
import os
import pickle
import sys

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

sys.path.append(r'/home/sverkip/data/BU-TD/yonathan/Recognicion/code/create_dataset')
class DataSetBase(Dataset):

    """
    Base class OmniglotDataSetBase.
    Supports initialization and get item methods.
    """

    def __init__(self, root: str, nclasses_existence: int, ndirections: int,is_train:bool, nexamples: int = None,
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
            filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(root) for f in fn]
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
      #  nexamples = 1000
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
    def __init__(self, root: str, opts:argparse, arg_and_head_index:int = 0, direction: tuple = (0, 0),is_train = True, nexamples: int = None, obj_per_row=6,
                 obj_per_col=1, split: bool = True):
        """
        Omniglot data-set.
        Args:
            root: Path to the data.
            opts: The model options.
            arg_and_head_index: The language index.
            direction: The direction tuple.
            nexamples: The number of examples.
            obj_per_row: Number of objects per row.
            obj_per_col: Number of columns per row.
            split: Whether to split the dataset.
        """

        super(DatasetAllDataSetTypes, self).__init__(root, nexamples, split, is_train=is_train)
        self.ntasks = opts.ntasks
        self.nclasses_existence = opts.nclasses[arg_and_head_index]
        self.direction = torch.tensor(direction)
        self.ndirections = opts.ndirections
        self.obj_per_row = obj_per_row
        self.obj_per_col = obj_per_col
        self.task_idx = torch.tensor(arg_and_head_index)
        self.edge_class = self.nclasses_existence

    def __getitem__(self, index):
        """
        Args:
            index: sample index.

        Returns: img, label_task, flag, label_all, label_existence.

        """
        # Getting root to the sample
      #  (direction_x, direction_y) = self.direction
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
        direction_type_ohe = torch.nn.functional.one_hot(self.direction, self.ndirections)
        # Getting the character embedding.
        char_type_one = torch.nn.functional.one_hot(torch.tensor(char), self.nclasses_existence)
        flag = torch.concat([direction_type_ohe, task_type_ohe, char_type_one], dim=0).float()
        # Concatenating into one flag.
        r, c = (label_all == char).nonzero()
        r, c = r[0], c[0]

        '''
        if 0 <= r + direction_y <= self.obj_per_col-1 and 0 <= c + direction_x <= self.obj_per_row-1:
            label_task = label_all[r + direction_x, c + direction_y]
        else:
            label_task = self.edge_class
        flag = torch.concat([direction_type_ohe, task_type_ohe, char_type_one], dim=0).float()
        '''

        if self.direction == 0:
            # right
            if c == (self.obj_per_row - 1):
                label_task = self.edge_class
            else:
                label_task = label_all[r, c + 1]

        if self.direction == 1:
            # left
            if c == 0:
                label_task = self.edge_class
            else:
                label_task = label_all[r, c - 1]

        elif self.direction == 2:
            # Up
            if r == (self.obj_per_col - 1):
                label_task = self.edge_class
            else:
                label_task = label_all[r + 1, c]

        elif self.direction == 3:
            # Down
            if r == 0:
                label_task = self.edge_class
            else:
                label_task = label_all[r - 1, c]

        # TODO change this to use conv2d with the proper filter
        label_existence, label_all, label_task = map(torch.tensor, (label_existence, label_all, label_task))
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
        label_task = self.calc_label_task_all(label_all, not_available_class).long()
        return img, label_task, flag, label_all, label_existence