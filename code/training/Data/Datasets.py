"""
Here we define the datasets, including guided and non-guided.
"""
import argparse
import pickle
from typing import Union, Tuple, Any

import torch
import torchvision.transforms
import torchvision.transforms as T
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from ..Utils import struct_to_input, tuple_direction_to_index
import os


class DataSetBase(Dataset):
    """
    Base class for all datasets.
    Supports initialization and get item methods(not used).
    """

    def __init__(self, root: str, nclasses: int, ndirections: int, is_train: bool, ntasks: int,
                 nexamples: Union[int, None] = None, obj_per_row: int = 6, obj_per_col: int = 1):
        """
        Args:
            root: The root to the data.
            nclasses:
            ndirections: The number of classes.
            nexamples: The number of classes.
            obj_per_row: The number of object per row.
            obj_per_col: The number of objects per column.
        """
        self.root: str = root  # The path to the data.
        self.nclasses: int = nclasses  # The number of classes.
        self.ntasks = ntasks
        self.ndirections: int = ndirections  # The number of directions.
        self.is_train: bool = is_train  # Is this a training set.
        self.nexamples: int = nexamples  # The number of examples.
        self.obj_per_row: int = obj_per_row  # The number of rows.
        self.obj_per_col: int = obj_per_col  # The number of columns.
        self.targets = [0 for _ in range(self.nexamples)]  # Used only for Avalanche_AI.
        self.split_size: int = 1000  # The split size we created the dataset according to.
        self.edge_class: Tensor = torch.tensor(nclasses)  # The edge class.

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
        folder_path = self.get_root_by_index(index=index)  # The path the sample directory
        data_path = os.path.join(folder_path, '%d_raw.pkl' % index)  # The path to the Sample itself.
        # Load the saved sample labels.
        with open(data_path, "rb") as new_data_file:
            raw_sample = pickle.load(new_data_file)
        return raw_sample

    def Compute_label_task(self, r: int, c: int, label_all: Tensor, direction: Tensor) -> Tensor:
        """
        Args:
            r: The row index.
            c: The column index.
            label_all: The label_all
            direction: The direction list.

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

    def __getitem__(self, index: int) -> tuple[Any, Any, Tensor, Tensor, Any, Any, Any, int]:
        root = self.get_root_by_index(index=index)  # The path to the sample.
        fname = os.path.join(root, '%d_img.jpg' % index)
        # Opening the image and converting to Tensor
        img = Image.open(fname).convert('RGB')  # Getting the image.
        img = T.ToTensor()(img)  # From Image to Tensor.
        img = 255 * img  # converting to RGB
        # Get raw sample.
        sample = self.get_raw_sample(index=index)
        # Converting the sample into input.
        label_existence, label_all, query_coord = struct_to_input(sample=sample)
        r, c = query_coord  # Getting the place we query about.
        char = label_all[r][c]  # Get the character we query about.
        # Getting the task embedding, telling which task we are solving now.
        # For emnist, fashion this is always 0 but for Omniglot it tells which language we use.
        # Getting the character embedding, which character we query about.
        char_type_one = torch.nn.functional.one_hot(char, self.nclasses).float()
        # Concatenating all three samples into one flag.
        sample_direction = sample.direction_query
        # If the task is part of the initial tasks, we solve all initial tasks together.
        return img, char_type_one, label_all, label_existence, r, c, sample_direction, index

    def __len__(self):
        """
        Returns: Length of the dataset.
        """
        return self.nexamples


class DatasetGuidedSingleTask(DataSetBase):
    """
    Guided Dataset.
    The guided dataset, returning query with argument.
    """

    def __init__(self, root: str, opts: argparse, nexamples: int, task_struct: Tuple[int, Tuple],
                 is_train=True, obj_per_row: int = 6, obj_per_col: int = 1):
        """

        Args:
            root: Path to the data.
            opts: The model options.
            nexamples: The number of examples.
            obj_per_row: Number of objects per row.
            obj_per_col: Number of objects per column.
        """
        super(DatasetGuidedSingleTask, self).__init__(ndirections=opts.data_obj.ndirections,
                                                      nclasses=opts.data_obj.nclasses[task_struct[0]], root=root,
                                                      nexamples=nexamples, is_train=is_train,
                                                      obj_per_col=obj_per_col, obj_per_row=obj_per_row,
                                                      ntasks=opts.data_obj.ntasks)
        self.opts = opts
        # The task index(not tuple).
        self.ds_type = opts.ds_type
        self.direction_tuple = torch.tensor(task_struct[-1])
        self.language_idx = torch.tensor(task_struct[0])  # The task id.
        self.edge_class = torch.tensor(self.nclasses)  # The 'border' class.
        self.transform = torchvision.transforms.Resize((128, 128))
        self.direction_idx, self.task_idx = tuple_direction_to_index(num_x_axis=self.opts.data_obj.num_x_axis,
                                             num_y_axis=self.opts.data_obj.num_y_axis,
                                             language_idx=self.language_idx,
                                             direction=self.direction_tuple, ndirections=self.opts.data_obj.ndirections)

    def __getitem__(self, index: int) -> tuple[Any, Tensor, Tensor, Tensor, Any, Tensor, Tensor]:
        """
        Args:
            index: sample index.

        Returns: img, label_task, flag, label_all, label_existence.

        """
        # Getting root to the sample
        img, char_type_one, label_all, label_existence, r, c, _, sample_id = \
            super(DatasetGuidedSingleTask, self).__getitem__(index=index)
        img = self.transform(img)
        label_task = self.Compute_label_task(r=r, c=c, label_all=label_all, direction=self.direction_tuple)
        return img, label_all, label_existence, label_task, char_type_one, self.direction_tuple, self.direction_idx, \
            self.language_idx, self.task_idx


class DatasetNonGuided(DatasetGuidedSingleTask):
    """
    Non-Guided Dataset.
    Needed for baselines.
    Returning for each character its adjacent character.
    """

    # In this dataset, we return for each character its adjacent character
    # according to the task to all characters.
    def Get_label_task_all(self, label_all: Tensor) -> Tensor:
        """
        Compute for each character its neighbor, if exists in the sample.
        Args:
            label_all: The label all flag.

        Returns: The label task.
        """

        label_adj_all = self.nclasses * torch.ones(size=(self.nclasses,), dtype=torch.long)
        for r, row in enumerate(label_all):  # Iterating over all rows.
            for c, char in enumerate(row):  # Iterating over all character in the row.
                # Compute the label task.
                res = self.Compute_label_task(r=r, c=c, label_all=label_all,
                                              direction=self.direction)
                label_adj_all[char] = res  # assign to the character.

        return label_adj_all

    def __getitem__(self, index: int) -> tuple[Any, Any, Any, Tensor, Any, Tensor, Tensor]:
        """
        Getting the sample with the 'return all' label task.
        Args:
            index: The index.

        Returns: The sample data.

        """
        img, label_all, label_existence, label_task, char_type_one, direction, task_idx = \
            super(DatasetNonGuided, self).__getitem__(index=index)
        # The same get item.
        # Change the label task to return all adjacent characters.
        label_task = self.Get_label_task_all(label_all=label_all)
        return img, label_all, label_existence, label_task, char_type_one, self.direction, self.task_idx
