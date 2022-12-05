import argparse
import os
import pickle
from typing import Union

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from Data_Creation.Create_dataset_classes import Sample
from training.Utils import tuple_direction_to_index

from Data_Creation.Create_dataset_classes import DsType

# Here we define the dataset classes.

class DataSetBase(Dataset):
    """
    Base class for all datasets.
    Supports initialization and get item methods(not used).
    """

    def __init__(self, root: str, nclasses_existence: int, ndirections: int, is_train: bool,
                 nexamples: Union[int, None] = None):
        """
        Args:
            root: The root to the Data_Creation.
            nclasses_existence:
            ndirections: The number of classes.
            nexamples: The number of classes.
        """
        self.root: str = root  # The path to the Data_Creation
        self.nclasses_existence: int = nclasses_existence  # The number of classes.
        self.ndirections: int = ndirections  # The number of directions.
        self.is_train: bool = is_train  # Is this a training set.
        self.nexamples: int = nexamples  # The number of examples.
        self.targets = [0 for _ in range(self.nexamples)]  # Used only for Avalanche_AI.
        self.split_size: int = 1000  # The split size we created the dataset according to.

    def get_root_by_index(self, index: int):
        """
        Given index returns the folder the sample is in.
        Args:
            index: The index.

        Returns: The path to the sample folder.

        """
        folder_path = os.path.join(self.root, '%d' % (index // self.split_size))  # The path to the Sample directory.
        return folder_path

    def get_raw_sample(self, index: int) -> pickle:
        """
        Get the raw sample in that index.
        Args:
            index: The sample index.

        Returns: The Sample.

        """
        folder_path = self.get_root_by_index(index)  # The path the sample directory
        data_path = os.path.join(folder_path, '%d_raw.pkl' % index)  # The path to the Sample itself.
        # Load the saved sample labels.
        with open(data_path, "rb") as new_data_file:
            raw_sample = pickle.load(new_data_file)
        return raw_sample

    def __getitem__(self, index: int):
        raise NotImplementedError

    def __len__(self):
        """
        Returns: Length of the dataset.
        """
        return self.nexamples


def struct_to_input(sample: Sample) -> tuple[torch, torch, tuple]:
    """
    Returning the sample attributed.
    Args:
        sample: Sample to return its attributes.

    Returns: The sample's attributes: label_existence, label_all ,flag, query coordinate.

    """
    # The label existence, telling for each entry whether the class exists or not.
    label_existence = sample.label_existence
    # All characters arranged.
    label_all = sample.label_ordered
    # The coordinate we query about.
    query_coord = sample.query_coord
    return label_existence, label_all, query_coord


class DatasetGuided(DataSetBase):
    def __init__(self, root: str, opts: argparse, nexamples: int, task_idx: int = 0, direction_tuple: tuple = (1, 0),
                 is_train=True, obj_per_row=6, obj_per_col=1):
        """
        Guided Dataset
        Args:
            root: Path to the Data_Creation.
            opts: The model options.
            nexamples: The number of examples.
            task_idx: The language index.
            direction_tuple: The direction tuple.
            obj_per_row: Number of objects per row.
            obj_per_col: Number of objects per column.
        """

        super(DatasetGuided, self).__init__(ndirections=opts.ndirections,
                                            nclasses_existence=opts.nclasses[task_idx], root=root,
                                            nexamples=nexamples, is_train=is_train)
        self.ntasks = opts.ntasks
        self.tuple_direction = direction_tuple
        # The direction index(not tuple).
        self.ds_type = opts.ds_type
        self.direction, _ = tuple_direction_to_index(num_x_axis=opts.num_x_axis, num_y_axis=opts.num_y_axis,
                                                     direction=direction_tuple,
                                                     ndirections=opts.ndirections, task_id=task_idx)
        self.obj_per_row = obj_per_row  # Number of row.
        self.obj_per_col = obj_per_col  # Number of columns.
        self.task_idx = torch.tensor(task_idx)  # The task id.
        self.edge_class = torch.tensor(self.nclasses_existence)  # The 'border' class.
        # TODO - GET RID OF THIS
        self.initial_tasks = opts.initial_directions  # The initial tasks.

    def Compute_label_task(self, r: int, c: int, label_all: np.ndarray, direction: tuple) -> torch:
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
        # Otherwise the target is 'border'.
        else:
            label_task = self.edge_class
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
        # Get raw sample.
        sample = self.get_raw_sample(index)
        # Converting the sample into input.
        label_existence, label_all, query_coord = struct_to_input(sample)
        r, c = query_coord  # Getting the place we query about.
        char = label_all[r][c]  # Get the character we query about.
        # Getting the task embedding, telling which task we are solving now.
        # For emnist,fashion this is always 0 but for Omniglot it tells which language we use.
        task_type_ohe = torch.nn.functional.one_hot(self.task_idx, self.ntasks)
        # Getting the direction embedding, telling which direction we solve now.
        direction_type_ohe = torch.nn.functional.one_hot(self.direction, self.ndirections)
        # Getting the character embedding, which character we query about.
        char_type_one = torch.nn.functional.one_hot(char, self.nclasses_existence)
        # Concatenating all three flags into one flag.
        flag = torch.concat([direction_type_ohe, task_type_ohe, char_type_one], dim=0).float()
        # TODO - GET RID OF THIS.
        # If the task is part of the initial tasks, we solve all initial tasks together.
        if self.ds_type is not DsType.Omniglot and self.tuple_direction in self.initial_tasks:
            label_tasks = []
            for task in self.initial_tasks:  # Iterating over all tasks and get their label task.
                label_task = self.Compute_label_task(r=r, c=c, label_all=label_all, direction=task)
                label_tasks.append(label_task)
            label_task = torch.tensor(label_tasks)
        else:  # Return  a unique label task.
            label_task = self.Compute_label_task(r=r, c=c, label_all=label_all, direction=self.tuple_direction)
        return img, label_task, flag, label_all, label_existence


class DatasetNonGuided(DatasetGuided):
    # In this dataset, we return for each character its adjacent character according to the direction to all characters.
    def Get_label_task_all(self, label_all: torch) -> torch:
        """
        Compute for each character its neighbor, if exists in the sample.
        Args:
            label_all: The label all flag.

        Returns: The label task.
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
        Getting the sample with the 'return all' label task.
        Args:
            index: The index.

        Returns: The sample Data_Creation.

        """
        img, label_task, flag, label_all, label_existence = super().__getitem__(index)  # The same get item.
        # Change the label task to return all adjacent characters.
        label_task = self.Get_label_task_all(label_all=label_all)
        return img, label_task, flag, label_all, label_existence
