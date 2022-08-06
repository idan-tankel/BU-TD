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



class cyclic_inputs_to_strcut:
    def __init__(self,inputs,stage):
        img, label_all, label_existence,flag, flag_stage_1, flag_stage_2, flag_stage_3 ,label_task_stage_1, label_task_stage_2, label_task_stage_3  = inputs
        self.image = img
        self.label_all = label_all
        self.label_existence = label_existence
        self.general_flag = flag
        if stage == 0:
         self.label_task = label_task_stage_1
         self.flag = flag_stage_1
        if stage == 1:
         self.label_task = label_task_stage_2
         self.flag = flag_stage_2
        if stage == 2:
         self.label_task = label_task_stage_3
         self.flag = flag_stage_3

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

    def __init__(self, root: str, parser:argparse, embedding_idx: int,  nexamples: int = None, split: bool = False, mean_image: torch = None):
        """
        :param root: path to the data language
        :param nclasses_existence: #TODO-make an order in this class.
        :param num_languages: number of languages in the data-set.
        :param embedding_idx: the number of Task embedding
        :param nexamples: The number of samples in the data-set.
        :param split: Whether we split the data-set into folders.
        :param mean_image: the mean image.
        """
        super(OmniglotDatasetLabelSingleTaskRight, self).__init__(root, parser.nclasses[embedding_idx][0],parser.ntasks, nexamples, split,
                                                             mean_image)
        self.embedding_idx = embedding_idx
        self.parser = parser
        self.ntasks = parser.ntasks
        self.direction = 0
        self.obj_per_row = 6
        self.grit_size = parser.grit_size
        self.num_grits = int(np.ceil(224 / self.grit_size))

    def __getitem__(self, index):
        """

        Args:
            index:

        Returns:
        """
        # Getting root to the sample
        edge_class = self.nclasses_existence
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
        # Creating the flag including task and argument one hot embedding.
        lan_type_ohe = torch.nn.functional.one_hot(task_emb_id, self.ntasks)
        # TODO CHANGE TO SUPPORT ANY SIZE.
        char_type_one = torch.nn.functional.one_hot(torch.tensor(char),self.nclasses_existence)
        # Concatenating into one flag.
        general_flag = torch.concat([lan_type_ohe, char_type_one], dim=0).float()
        r,c = (label_all == char).nonzero()
        (r,c) = (r[0], c[0])
        character_location = sample.keypoints[c]
        # Getting the location and the character in the desired direction.
        if self.direction == 0:
            # right
            if c == (self.obj_per_row - 1):
                neighboor_location = [224,112]
                neighboor_character = edge_class
            else:
                neighboor_location = list(sample.keypoints[c + 1])
                neighboor_character = label_all[r,c + 1]

        if self.direction == 1:
            # left
            if c == 0:
                neighboor_location = [0,0]
                neighboor_character = edge_class
            else:
                neighboor_location = sample.keypoints[c-1]
                neighboor_character = label_all[r,c-1]

        label_existence, label_all, neighboor_location,character_location, id = map(torch.tensor, (label_existence, label_all, neighboor_location,character_location, id))
        label_existence = label_existence.float()
        #Creating the label_task and the flag for stage 1.

        flag_stage_1 = torch.nn.functional.one_hot(torch.tensor(char),self.nclasses_existence ).float() # The character we are working on.
        label_task_stage_1 = character_location.long() # The location of the character on the image.
        label_task_stage_1 = torch.ceil(torch.div(label_task_stage_1, self.grit_size)).long() # The location of the character on the grit.
        # Creating the label_task and the flag for stage 2.
        (x,y) = torch.ceil(torch.div(character_location, self.grit_size)).long() # getting the location of the character on the grit.
        flag_stage_2_x = torch.nn.functional.one_hot(x, self.num_grits) # Making one hot embedding of the x coordinate.
        flag_stage_2_y = torch.nn.functional.one_hot(y, self.num_grits) # Making one hot embedding of the y coordinate.
        flag_stage_2 = torch.concat([flag_stage_2_x, flag_stage_2_y], dim=0).float() # Getting one flag.
        label_task_stage_2 = torch.div(neighboor_location, self.grit_size).long() # The label task is the location of the neighbor on the grit.
        # Creating the label_task and the flag for stage 3.
        (x_location_neightboor, y_location_neighboor) = torch.div(neighboor_location, self.grit_size).long() # getting the location of the neighbor on the grit.
        flag_stage_3_x = torch.nn.functional.one_hot(x_location_neightboor, self.num_grits) # Making one hot embedding of the x coordinate.
        flag_stage_3_y = torch.nn.functional.one_hot(y_location_neighboor, self.num_grits ) # Making one hot embedding of the y coordinate.
        flag_stage_3 = torch.concat([ flag_stage_3_x, flag_stage_3_y], dim=0).float() # Making a one flag.
        label_task_stage_3 = neighboor_character # The label task of stage 3 is the neighbor character.
        return  img, label_all, label_existence,general_flag, flag_stage_1, flag_stage_2, flag_stage_3 ,label_task_stage_1, label_task_stage_2, label_task_stage_3



