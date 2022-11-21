import argparse
import os
import pickle
from typing import Union

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from data.utils.Create_dataset_classes import Sample
from training.Utils import tuple_direction_to_index


class DataSetBase(Dataset):
    """
    Base class for all datasets.
    Supports initialization and get item methods(not used).
    """

    def __init__(self, root: str, nclasses_existence: int, ndirections: int, is_train: bool,
                 nexamples: Union[int, None] = None):
        """
        Args:
            root: The root to the data.
            nclasses_existence:
            ndirections: The number of classes.
            nexamples: The number of classes.
        """
        self.root = root  # The path to the data
        self.nclasses_existence = nclasses_existence  # The number of classes.
        self.ndirections = ndirections  # The number of directions.
        self.is_train = is_train  # Is this a training set.
        self.split_size = 1000
        if nexamples is None:  # Getting all samples if the number is not given(None).
            filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(root) for f in fn]
            images = [f for f in filenames if f.endswith('_img.jpg')]
            self.nexamples = len(images)
        else:
            self.nexamples = nexamples
     #   self.nexamples = 300
        self.targets = [0 for _ in range(self.nexamples)]  # Used only for Avalanche_AI.

    def get_root_by_index(self, index: int):
        """
        Given index returns the folder the sample is in.
        Args:
            index: The index.

        Returns: The path to the sample folder.

        """
        folder_path = os.path.join(self.root, '%d' % (index // self.split_size))  # The path to the Sample.
        return folder_path

    def get_raw_sample(self, index: int) -> pickle:
        """
        Get the raw sample in that index.
        Args:
            index: The sample index.

        Returns: The Sample.

        """
        folder_path = self.get_root_by_index(index)
        data_path = os.path.join(folder_path, '%d_raw.pkl' % index)  # The path to the Sample.
        with open(data_path, "rb") as new_data_file:
            raw_sample = pickle.load(new_data_file)
        return raw_sample

    def __getitem__(self, index: int):
        """
        Given index return a sample.
        Override by all classes.
        Args:
            index: The index.

        Returns: Raw Sample.

        """
        return self.get_raw_sample(index)

    def __len__(self):
        """
        Returns: Length of the dataset.
        """
        return self.nexamples


def struct_to_input(sample: Sample) -> tuple[torch]:
    """
    From sample to inputs.
    Args:
        sample: Sample to return its attributes.

    Returns: The sample's attributes: label_existence, label_all ,flag, label_task.

    """
    label_existence = torch.tensor(sample.label_existence).float()
    label_task = torch.tensor(sample.label_task)
    label_all = sample.label_ordered
    flag = sample.flag
    return label_existence, label_all, flag, label_task


class DatasetGuided(DataSetBase):
    def __init__(self, root: str, opts: argparse, task_idx: int = 0, direction: tuple = (1, 0), is_train=True,
                 nexamples: int = None, obj_per_row=6,
                 obj_per_col=1):
        """
        Guided Dataset
        Args:
            root: Path to the data.
            opts: The model options.
            task_idx: The language index.
            direction: The direction tuple.
            nexamples: The number of examples.
            obj_per_row: Number of objects per row.
            obj_per_col: Number of columns per row.
        """

        super(DatasetGuided, self).__init__(ndirections=opts.ndirections,
                                            nclasses_existence=opts.nclasses[task_idx], root=root,
                                            nexamples=nexamples, is_train=is_train)
        self.ntasks = opts.ntasks
        self.tuple_direction = direction
        # The direction index(not tuple).
        self.direction, _ = tuple_direction_to_index(opts.num_x_axis, opts.num_y_axis, direction,
                                                     ndirections=opts.ndirections, task_id=task_idx)
        self.obj_per_row = obj_per_row  # Number of row.
        self.obj_per_col = obj_per_col  # Number of columns.
        self.task_idx = torch.tensor(task_idx)  # The task id.
        self.edge_class = self.nclasses_existence  # The 'border' class.
        self.initial_tasks = opts.initial_tasks  # The initial tasks.
        self.trans = T.Resize((112, 224))

    def Compute_label_task(self, r: int, c: int, label_all: np.ndarray, direction: tuple) -> int:
        """
        Args:
            r: The row index.
            c: The column index.
            label_all: The label_all
            direction: The direction.

        Returns: The label task.

        """
        direction_x, direction_y = direction
        # If the target not in the Border.
        if 0 <= r + direction_y <= self.obj_per_col - 1 and 0 <= c + direction_x <= self.obj_per_row - 1:
            label_task = label_all[r + direction_y, c + direction_x]
        else:
            label_task = self.edge_class  # target is the border otherwise.
        return label_task

    def __getitem__(self, index: int) -> tuple[torch]:
        """
        Args:
            index: sample index.

        Returns: img, label_task, flag, label_all, label_existence.

        """
        # Getting root to the sample
        root = self.get_root_by_index(index)  # The path to the sample.
        fname = os.path.join(root, '%d_img.jpg' % index)
        # Opening the image and converting to Tensor
        img = Image.open(fname).convert('RGB')  # Getting the image.
        img = T.ToTensor()(img)  # From Image to Tensor.
        img = 255 * img  # converting to RGB
        img = self.trans(img)
        # Get raw sample.
        sample = self.get_raw_sample(index)
        # Converting the sample into input.
        label_existence, label_all, flag, label_task = struct_to_input(sample)
        char = flag[1].item()  # The referred character
        # Getting the task embedding, telling which task we are solving now.
        # For emnist,fashion this is always 0 but for Omniglot it tells which language we use.
        task_type_ohe = torch.nn.functional.one_hot(self.task_idx, self.ntasks)
        # Getting the direction embedding, telling which direction we solve now.
        direction_type_ohe = torch.nn.functional.one_hot(self.direction, self.ndirections)
        # Getting the character embedding, which character we look for.
        char_type_one = torch.nn.functional.one_hot(torch.tensor(char), self.nclasses_existence)
        # Concatenating all three flags into one flag.
        flag = torch.concat([direction_type_ohe, task_type_ohe, char_type_one], dim=0).float()
        r, c = (label_all == char).nonzero()  # Getting the place we query about.
        r, c = r[0], c[0]
        # If the task is part of the initial tasks, we solve all tasks together.
        if self.tuple_direction in self.initial_tasks:
            label_tasks = []
            for task in self.initial_tasks:  # Iterating over all tasks and get their label task.
                label_task = self.Compute_label_task(r=r, c=c, label_all=label_all, direction=task)
                label_tasks.append(label_task)
            label_task = torch.tensor(label_tasks)
        else:  # Return  a unique label task.
            label_task = torch.tensor(
                self.Compute_label_task(r=r, c=c, label_all=label_all, direction=self.tuple_direction))
        label_all = torch.tensor(label_all)
        label_task = label_task.long()
        return img, label_task, flag, label_all, label_existence


class DatasetNonGuided(DatasetGuided):
    # In this dataset, we return adjacent to all characters.
    def calc_label_task_all(self, label_all: torch) -> torch:
        """
        Compute for each character its neighbor if exists in the sample.
        Args:
            label_all: The label all flag.

        Returns: The label all task.
        """

        label_adj_all = self.nclasses_existence * torch.ones(self.nclasses_existence)
        for r, row in enumerate(label_all):  # Iterating over all rows.
            for c, char in enumerate(row):  # Iterating over all character in the row.
                res = self.Compute_label_task(r=r, c=c, label_all=label_all,
                                              direction=self.tuple_direction)  # Compute the label task.
                label_adj_all[char] = res

        return label_adj_all

    def __getitem__(self, index: int) -> tuple[torch]:
        """
        Getting the sample with the all label task.
        Args:
            index: The index.

        Returns: The sample data.

        """
        img, label_task, flag, label_all, label_existence = super().__getitem__(index)  # The same get item.
        # Change the label task to return all adjacent characters.
        label_task = self.calc_label_task_all(
            label_all=label_all)
        return img, label_task, flag, label_all, label_existence  # Return all sample.
