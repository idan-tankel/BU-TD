from PIL import Image
import pickle
import os
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import argparse
from supp.FlagAt import *
from supp.general_functions import *


def get_omniglot_dictionary(opts: argparse, raw_data_folderpath: str) -> list:
    """
    :param opts:
    :param raw_data_folderpath:
    :return: list of nclasses per language.
    """
    nclasses_dictionary = {}
    dictionary = create_dict(raw_data_folderpath)
    initial_tasks = opts.initial_tasks
    nclasses_dictionary[0] = sum(
        dictionary[task] for task in initial_tasks)  # receiving number of characters in the initial tasks.
    nclasses = []
    if opts.model_flag is FlagAt.NOFLAG:  # If flag = NOFLAG all the characters must be recognized(usually 6) otherwise only one character.
        num_chars = args.nargs
    else:
        num_chars = 1
    for i in range(opts.ntasks - 1):  # copying the number of characters for all classes
        nclasses_dictionary[i + 1] = dictionary[i]
    for i in range(opts.ntasks):  # creating nclasses according to the dictionary and num_chars
        nclasses.append([nclasses_dictionary[i] for _ in range(num_chars)])
    return nclasses


class OmniglotDataSetBase(data.Dataset):
    """
    Base class OmniglotDataSetBase.
    Supports initialization and get item methods.
    """

    def __init__(self, root: str, nclasses_existence: int, ndirections: int, nexamples: int = None, split: bool = False,
                 mean_image: torch = None) -> None:
        """
        initializes the DataSet by root to the data
        :param root:root to the data
        :param nclasses_existence:
        :param ndirections:
        :param nexamples:
        :param split:
        :param mean_image:
        """
        self.root = root
        self.nclasses_existence = nclasses_existence
        self.ndirections = ndirections
        self.mean_image = mean_image
        self.split = split
        self.splitsize = 1000
        if nexamples is None:
            filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(root) for f in fn]
            images = [f for f in filenames if f.endswith('_img.jpg')]
            self.nexamples = len(images)
        else:
            self.nexamples = nexamples

    def get_root_by_index(self, index: int) -> str:
        """
        :param index: index to retrieve from
        :return: path to the sample.
        """
        if self.split:
            root = os.path.join(self.root, '%d' % (index // self.splitsize))
        else:
            root = self.root
        return root

    def get_raw_sample(self, index):
        """
        :param index: index to retrieve from.
        :return: raw_sample.
        """
        root = self.get_root_by_index(index)
        data_fname = os.path.join(root, '%d_raw.pkl' % index)
        with open(data_fname, "rb") as new_data_file:
            raw_sample = pickle.load(new_data_file)
        return raw_sample

    def __len__(self):
        """
        :return: length of the dataset.
        """
        return self.nexamples


class inputs_to_struct:
    def __init__(self, inputs: torch) -> None:
        """
        Creates a class that stores the input as a struct
        :param inputs:  img, label_all, label_task, flag.
        """
        img, label_all, label_existence, label_task, flag = inputs
        self.image = img
        self.label_all = label_all
        self.label_existence = label_existence
        self.label_task = label_task
        self.flag = flag

def struct_to_input(sample: object) -> tuple:
    """
    :param sample:
    :return: Given sample return its attributes including label_existence, label_all ,flag, label_task, id.
    """
    label_existence = sample.label_existence
    label_all = sample.label_ordered
    flag = sample.flag
    label_task = sample.label_task
    id = sample.id
    return label_existence, label_all, flag, label_task, id


class OmniglotDatasetLabelSingleTask(OmniglotDataSetBase):
    """
    The most used dataset.
    Returns the sample needed for the guided version of the BU-TD model.
    The goal is given img,index idx to return img[index].
    """

    def __init__(self, root: str, nclasses_existence: int, num_languages: int, embedding_idx: int, nargs: int,
                 nexamples: int = None, split: bool = False, mean_image: torch = None) -> None:
        """
        :param root: path to the data language
        :param nclasses_existence: #TODO-make an order in this class.
        :param num_languages: number of characters in the language
        :param embedding_idx: the number of Task embedding
        :param nexamples:
        :param split:
        :param nargs:
        :param mean_image:
        """
        super(OmniglotDatasetLabelSingleTask, self).__init__(root, nclasses_existence, num_languages, nexamples, split,
                                                             mean_image)
        self.embedding_idx = embedding_idx
        self.num_languages = num_languages
        self.nargs = nargs

    def __getitem__(self, index):
        # Getting root to the sample
        root = self.get_root_by_index(index)
        fname = os.path.join(root, '%d_img.jpg' % index)
        # Opening the image and converting to Tensor
        img = Image.open(fname).convert('RGB')
        img = transforms.ToTensor()(img)
        img = 255 * img  # converting to RGB
        # subtracting the mean if mean_image is not None and converting to float
        if self.mean_image is not None:
            img -= self.mean_image
            img = img.float()
        # get raw sample
        sample = self.get_raw_sample(index)
        # Converting the sample into input.
        (label_existence, label_all, flag, label_task, id) = struct_to_input(sample)
        label_existence, label_all, label_task, id = map(torch.tensor, (label_existence, label_all, label_task, id))
        char = flag[1].item()  # The referred character
        # Converting the emb id into tensor.
        task_emb_id = torch.tensor(self.embedding_idx)
        char_id = ((label_all == char).nonzero(as_tuple=False)[0][1])
        # Creating the flag including task and argument one hot embedding.
        lan_type_ohe = torch.nn.functional.one_hot(task_emb_id, self.num_languages)
        char_type_one = torch.nn.functional.one_hot(char_id, self.nargs)
        # Concatenating into one flag.
        flag = torch.concat([lan_type_ohe, char_type_one], dim=0).float()
        label_task = label_task.view((-1))
        return img, label_all, label_existence, label_task, flag


# TODO-make an order in this class.

class OmniglotDatasetLabelAll(OmniglotDataSetBase):
    def __init__(self, root, nclasses_existence, ndirections, tasks, nexamples=None, split=False, mean_image=None):
        super(OmniglotDatasetLabelAll, self).__init__(root, nclasses_existence, ndirections, nexamples,
                                                      split, mean_image)
        self.tasks = tasks
        self.ntasks = len(tasks)
        self.BS = 20

    def get_root_by_index(self, index):
        i = (index // self.BS) % self.ntasks
        idx = index // (self.ntasks * self.BS)
        j = (idx * self.BS + index % self.BS) % len(self.tasks[i])
        return i, j

    def __getitem__(self, index):
        (i, j) = self.get_root_by_index(index)
        return self.tasks[i][j]

    def __len__(self):
        return max([len(self.tasks[i]) for i in range(len(self.tasks))]) * self.ntasks


# TODO-make an order in this class.

class OmniglotDatasetLabelSingleTaskRight(OmniglotDataSetBase):
    """
    The most used dataset.
    Returns the sample needed for the guided version of the BU-TD model.
    The goal is given img,index idx to return img[index].
    """

    def __init__(self, root: str, nclasses_existence: int, num_languages: int, embedding_idx: int, nargs: int,
                 nexamples: int = None, split: bool = False, mean_image: torch = None) -> None:
        """
        :param root: path to the data language
        :param nclasses_existence: #TODO-make an order in this class.
        :param num_languages: number of characters in the language
        :param embedding_idx: the number of Task embedding
        :param nexamples:
        :param split:
        :param nargs:
        :param mean_image:
        """
        super(OmniglotDatasetLabelSingleTaskRight, self).__init__(root, nclasses_existence, num_languages, nexamples, split,
                                                             mean_image)
        self.embedding_idx = embedding_idx
        self.num_languages = num_languages
        self.nargs = nargs
        self.nclasses_existence = nclasses_existence

    def __getitem__(self, index):
        # Getting root to the sample
        root = self.get_root_by_index(index)
        fname = os.path.join(root, '%d_img.jpg' % index)
        # Opening the image and converting to Tensor
        img = Image.open(fname).convert('RGB')
        img = transforms.ToTensor()(img)
        img = 255 * img  # converting to RGB
        # subtracting the mean if mean_image is not None and converting to float
        if self.mean_image is not None:
            img -= self.mean_image
            img = img.float()
        # get raw sample
        sample = self.get_raw_sample(index)
        # Converting the sample into input.
        (label_existence, label_all, flag, label_task, id) = struct_to_input(sample)
        char = flag[1].item()  # The referred character
        # Converting the emb id into tensor.
        task_emb_id = torch.tensor(self.embedding_idx)
    #    char_id = ((label_all == char).nonzero(as_tuple=False)[0][1])
        # Creating the flag including task and argument one hot embedding.
        lan_type_ohe = torch.nn.functional.one_hot(task_emb_id, self.num_languages)
        # TODO CHANGE TO SUPPORT ANY SIZE.
        char_type_one = torch.nn.functional.one_hot(torch.tensor(char),240 )
        # Concatenating into one flag.
        flag = torch.concat([lan_type_ohe, char_type_one], dim=0).float()
       # adj_type, char = flag
        adj_type = 1
        obj_per_row = 6
        edge_class = self.nclasses_existence
        r,c = (label_all == char).nonzero()

        if adj_type == 0:
            # right
            if c == (obj_per_row - 1):
                label_task = edge_class
            else:
                label_task = label_all[r, c + 1]
        if adj_type == 1:
            # left
            if c == 0:
                label_task = edge_class
            else:
                label_task = label_all[r, c - 1]
        c = c[0]
        label_existence, label_all, label_task, id = map(torch.tensor, (label_existence, label_all, label_task, id))
        label_existence = label_existence.float()
     #   label_task = torch.tensor(label_task).view(-1)
        label_task = label_task.view([-1])
        label_task = torch.tensor(sample.keypoints[c]).type(torch.LongTensor)//4
       # label_task = torch.tensor(sample.keypoint)
        return img, label_all, label_existence, label_task, flag
