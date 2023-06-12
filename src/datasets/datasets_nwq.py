import pickle
import sys
import yaml
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import git
from Configs.Config import Config
from typing import Union
from torchvision.datasets import VisionDataset


class DatasetInfo(VisionDataset):
    r"""
    datasetForBUTD is a class to represent a dataset that is good to BU-TD type of models 
    (i.e, has some sort of image and set of tasks to look within it, which can be sequential or not)

    This class gives an interface to the dataset, which is later on used by the dataloader to load some saved attributes.

    In this scenerio, the dataset is first created by a torchvision.dataset, and then is saved to a folder, and then procecced via the process_data function.
    Since we do not want the old __getitem__ of the dataset, we would give up on it and leave the original torchvision.dataset as is, and create a new one that will be used by the dataloader.
    """

    def __init__(self, regular_dataset: VisionDataset) -> None:
        self.original_dataset_object = regular_dataset

    def __getitem__(self, index: int):
        """
        Args:
            index: The sample index.

        Returns: The Sample with all the important attributes as well as enrichments
        Since the model is map like (uses the filetree as a good enumeration)
        """
        # temporal - uses the original dataset as an interface
        return self.original_dataset_object[index]

    def __len__(self):
        """
        __len__ The number of examples in the dataset. Since each example is processed, the length is the same as the original dataset

        Returns:
            `int`: The number of examples in the dataset.
        """
        return len(self.original_dataset_object)

    def process_dataset(self):
        """
        process_dataset That function iterates all the data in `self.original_dataset_object` and adds a json file to each sample.
        This function uses the root directory of the original dataset
        """
        pass
