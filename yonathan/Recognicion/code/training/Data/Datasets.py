import argparse
import os
import pickle
from typing import Union

import torch
from torch import Tensor
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from training.Utils import tuple_direction_to_index, struct_to_input


# Here we define the dataset classes.

class DataSetBase(Dataset):
    """
    Base class for all datasets.
    Supports initialization and get item methods(not used).
    """

    def __init__(self, root: str, nclasses: int, ndirections: int, is_train: bool,
                 nexamples: Union[int, None] = None, obj_per_row=6, obj_per_col=1):
        """
        Args:
            root: The root to the data.
            nclasses:
            ndirections: The number of classes.
            nexamples: The number of classes.
        """
        self.root: str = root  # The path to the data.
        self.nclasses: int = nclasses  # The number of classes.
        self.ndirections: int = ndirections  # The number of directions.
        self.is_train: bool = is_train  # Is this a training set.
        self.nexamples: int = nexamples  # The number of examples.
        self.obj_per_row: int = obj_per_row  # The number of rows.
        self.obj_per_col: int = obj_per_col  # The number of columns.
      #  self.nexamples = 200
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
        folder_path = self.get_root_by_index(index=index)  # The path the sample directory
        data_path = os.path.join(folder_path, '%d_raw.pkl' % index)  # The path to the Sample itself.
        # Load the saved sample labels.
        with open(data_path, "rb") as new_data_file:
            raw_sample = pickle.load(new_data_file)
        return raw_sample

    def Compute_label_task(self, r: int, c: int, label_all: Tensor, direction_list: list[tuple]) -> Tensor:
        """
        Args:
            r: The row index.
            c: The column index.
            label_all: The label_all
            direction_list: The direction.

        Returns: The label task.

        """
        labels_task = []
        for direction in direction_list:
            direction_x, direction_y = direction
            # If the target not in the Border.
            if 0 <= r + direction_y <= self.obj_per_col - 1 and 0 <= c + direction_x <= self.obj_per_row - 1:
                label_task = label_all[r + direction_y, c + direction_x]
            # Otherwise the target is 'border'.
            else:
                label_task = self.edge_class
            labels_task.append(label_task)
        labels_task = torch.tensor(labels_task)
        return labels_task

    def __getitem__(self, index: int):
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
        # For emnist,fashion this is always 0 but for Omniglot it tells which language we use.
        # Getting the character embedding, which character we query about.
        char_type_one = torch.nn.functional.one_hot(char, self.nclasses)
        # Concatenating all three flags into one flag.
        flag = char_type_one
        sample_direction = sample.direction_query
        # If the task is part of the initial tasks, we solve all initial tasks together.
        return img, flag, label_all, label_existence, r, c, sample_direction

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

    def __init__(self, root: str, opts: argparse, nexamples: int, task_idx: int = 0,
                 direction_tuple: list[tuple] = [(1, 0)],
                 is_train=True, obj_per_row=6, obj_per_col=1):
        """

        Args:
            root: Path to the data.
            opts: The model options.
            nexamples: The number of examples.
            task_idx: The language index.
            direction_tuple: The direction tuple.
            obj_per_row: Number of objects per row.
            obj_per_col: Number of objects per column.
        """

        super(DatasetGuidedSingleTask, self).__init__(ndirections=opts.ndirections,
                                                      nclasses=opts.nclasses[task_idx], root=root,
                                                      nexamples=nexamples, is_train=is_train,
                                                      obj_per_col=obj_per_col, obj_per_row=obj_per_row)
        self.ntasks = opts.ntasks
        self.tuple_direction = direction_tuple
        # The direction index(not tuple).
        self.ds_type = opts.ds_type
        self.direction, _ = tuple_direction_to_index(num_x_axis=opts.num_x_axis, num_y_axis=opts.num_y_axis,
                                                     direction=direction_tuple[0],
                                                     ndirections=opts.ndirections, task_id=task_idx)
        self.task_idx = torch.tensor(task_idx)  # The task id.
        self.edge_class = torch.tensor(self.nclasses)  # The 'border' class.

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            index: sample index.

        Returns: img, label_task, flag, label_all, label_existence.

        """
        # Getting root to the sample
        img, char_type_one, label_all, label_existence, r, c, _ = super().__getitem__(index=index)
        task_type_ohe = torch.nn.functional.one_hot(self.task_idx, self.ntasks)
        # Getting the direction embedding, telling which direction we solve now.
        direction_type_ohe = torch.nn.functional.one_hot(self.direction, self.ndirections)
        # Getting the character embedding, which character we query about.
        # Concatenating all three flags into one flag.
        flag = torch.concat([direction_type_ohe, task_type_ohe, char_type_one], dim=0).float()
        # If the task is part of the initial tasks, we solve all initial tasks together.
        label_task = self.Compute_label_task(r=r, c=c, label_all=label_all, direction_list=self.tuple_direction)
        return img, label_task, flag, label_all, label_existence


class DatasetGuidedInterleaved(DataSetBase):
    """
    Guided Dataset.
    The guided dataset, returning query with argument.
    """

    def __init__(self, root: str, opts: argparse, nexamples: int, task_idx: int = 0,
                 is_train=True, obj_per_row=6, obj_per_col=1):
        """

        Args:
            root: Path to the data.
            opts: The model options.
            nexamples: The number of examples.
            task_idx: The language index.
            direction_tuple: The direction tuple.
            obj_per_row: Number of objects per row.
            obj_per_col: Number of objects per column.
        """

        super(DataSetBase, self).__init__(ndirections=opts.ndirections,
                                          nclasses=opts.nclasses[task_idx], root=root,
                                          nexamples=nexamples, is_train=is_train)
        self.ntasks = opts.ntasks
        # The direction index(not tuple).
        self.ds_type = opts.ds_type
        self.obj_per_row = obj_per_row  # Number of row.
        self.obj_per_col = obj_per_col  # Number of columns.
        self.task_idx = torch.tensor(task_idx)  # The task id.
        self.edge_class = torch.tensor(self.nclasses)  # The 'border' class.

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            index: sample index.

        Returns: img, label_task, flag, label_all, label_existence.

        """
        # Getting root to the sample
        img, char_type_one, label_all, label_existence, r, c, sample_direction = super().__getitem__(index=index)
        task_type_ohe = torch.nn.functional.one_hot(self.task_idx, self.ntasks)
        # Getting the direction embedding, telling which direction we solve now.
        direction_type_ohe = torch.nn.functional.one_hot(sample_direction, self.ndirections)
        # Getting the character embedding, which character we query about.
        # Concatenating all three flags into one flag.
        flag = torch.concat([direction_type_ohe, task_type_ohe, char_type_one], dim=0).float()
        # If the task is part of the initial tasks, we solve all initial tasks together.
        label_task = self.Compute_label_task(r=r, c=c, label_all=label_all, direction_list=[sample_direction])
        return img, label_task, flag, label_all, label_existence


class DatasetAvatar(DataSetBase):
    """
    Guided Dataset.
    The guided dataset, returning query with argument.
    """

    def __init__(self, root: str, opts: argparse, nexamples: int, task_idx: int = 0,
                 direction_tuple: list[tuple] = [(1, 0)],
                 is_train=True, obj_per_row=6, obj_per_col=1):
        """

        Args:
            root: Path to the data.
            opts: The model options.
            nexamples: The number of examples.
            task_idx: The language index.
            direction_tuple: The direction tuple.
            obj_per_row: Number of objects per row.
            obj_per_col: Number of objects per column.
        """

        super(DatasetAvatar, self).__init__(ndirections=opts.ndirections,
                                            nclasses=opts.nclasses[task_idx], root=root,
                                            nexamples=nexamples, is_train=is_train,
                                            obj_per_col=obj_per_col, obj_per_row=obj_per_row)
        self.ntasks = opts.ntasks
        self.tuple_direction = direction_tuple
        # The direction index(not tuple).
        self.ds_type = opts.ds_type
        self.direction, _ = tuple_direction_to_index(num_x_axis=opts.num_x_axis, num_y_axis=opts.num_y_axis,
                                                     direction=direction_tuple[0],
                                                     ndirections=opts.ndirections, task_id=task_idx)
        self.task_idx = torch.tensor(task_idx)  # The task id.
        self.edge_class = torch.tensor(self.nclasses)  # The 'border' class.

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            index: sample index.

        Returns: img, label_task, flag, label_all, label_existence.

        """
        # Getting root to the sample
        sample = self.get_raw_sample(index=index)
        root = self.get_root_by_index(index=index)  # The path to the sample.
        fname = os.path.join(root, '%d_img.jpg' % index)
        # Opening the image and converting to Tensor
        img = Image.open(fname).convert('RGB')  # Getting the image.
        img = T.ToTensor()(img)  # From Image to Tensor.
        label_all, label_existence = sample.label_ordered, sample.label_existence
        task_type_ohe = torch.nn.functional.one_hot(self.task_idx, self.ntasks)
        # Getting the direction embedding, telling which direction we solve now.
        direction_type_ohe = torch.nn.functional.one_hot(self.direction, self.ndirections)
        # Getting the character embedding, which character we query about.
        # Concatenating all three flags into one flag.
        #   print(sample.query_part_id)
        #  print(label_all.shape)
        label_all = label_all.view([-1, 7])
        query = sample.query_part_id
        # label_existence = torch.concat([label_existence, torch.zeros(2,)],dim=1)
        char_type_one = torch.nn.functional.one_hot(label_all[query][0], 8)
        flag = torch.concat([direction_type_ohe, task_type_ohe, char_type_one], dim=0).float()
        # If the task is part of the initial tasks, we solve all initial tasks together.
        label_task = label_all[query, -3]
        #  print(label_task)
        return img, label_task, flag, label_all, label_existence


class DatasetNonGuided(DatasetGuidedSingleTask):
    """
    Non-Guided Dataset.
    Returning for each character its adjacent character.
    """

    # In this dataset, we return for each character its adjacent character according to the direction to all characters.
    def Get_label_task_all(self, label_all: Tensor) -> Tensor:
        """
        Compute for each character its neighbor, if exists in the sample.
        Args:
            label_all: The label all flag.

        Returns: The label task.
        """

        label_adj_all = self.nclasses * torch.ones(self.nclasses, dtype=int)
        for r, row in enumerate(label_all):  # Iterating over all rows.
            for c, char in enumerate(row):  # Iterating over all character in the row.
                res = self.Compute_label_task(r=r, c=c, label_all=label_all,
                                              direction_list=self.tuple_direction)  # Compute the label task.
                label_adj_all[char] = res  # assign to the character.

        return label_adj_all

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Getting the sample with the 'return all' label task.
        Args:
            index: The index.

        Returns: The sample data.

        """
        img, label_task, flag, label_all, label_existence = super().__getitem__(index=index)  # The same get item.
        # Change the label task to return all adjacent characters.
        label_task = self.Get_label_task_all(label_all=label_all)
        return img, label_task, flag, label_all, label_existence
