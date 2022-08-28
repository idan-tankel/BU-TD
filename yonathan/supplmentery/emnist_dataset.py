import os
import pickle
import random  # torch random transforms uses random
from types import SimpleNamespace

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


class EMNISTAdjDatasetBase(data.Dataset):
    def __init__(self, root, nclasses_existence,ndirections, nexamples = None, split = False,mean_image = None):
        self.root = root
        self.nclasses_existence = nclasses_existence
        self.ndirections = ndirections
        self.mean_image = mean_image
        self.split = split
        self.splitsize=1000
        if nexamples is None:
            # just in order to count the number of examples

            # all files recursively
            filenames=[os.path.join(dp, f) for dp, dn, fn in os.walk(root) for f in fn]

            images = [f for f in filenames if f.endswith('_img.jpg')]
            self.nexamples = len(images)
        else:
            self.nexamples = nexamples

    def get_root_by_index(self, index):
        if self.split:
            root= os.path.join(self.root,'%d' % (index//self.splitsize))
        else:
            root= self.root
        return root

    def get_base_raw_sample(self, index):
        root = self.get_root_by_index(index)
        data_fname=os.path.join(root,'%d_raw.pkl' % index)
        with open(data_fname, "rb") as new_data_file:
            raw_sample = pickle.load(new_data_file)
        fname=os.path.join(root,'%d_img.jpg' % index)
        img = Image.open(fname).convert('RGB')
        fname=os.path.join(root,'%d_seg.jpg' % index)
        seg = Image.open(fname).convert('RGB')
        raw_sample.img = img
        raw_sample.seg = seg
        return raw_sample

    def get_raw_sample(self, index):
        root = self.get_root_by_index(index)
        data_fname=os.path.join(root,'%d_raw.pkl' % index)
        with open(data_fname, "rb") as new_data_file:
            raw_sample = pickle.load(new_data_file)
        return raw_sample

    def __len__(self):
        return self.nexamples

class EMNISTAdjDataset(EMNISTAdjDatasetBase):
    def __getitem__(self, index):
        root = self.get_root_by_index(index)
        fname=os.path.join(root,'%d_img.jpg' % index)
        img = Image.open(fname).convert('RGB')
        fname=os.path.join(root,'%d_seg.jpg' % index)
        seg = Image.open(fname).convert('RGB')
        img,seg = map(
                transforms.ToTensor(), (img,seg))
        img,seg = 255*img,255*seg
        if self.mean_image is not None:
            img -= self.mean_image
            img=img.float()
        sample = self.get_raw_sample(index)
        label_existence = sample.label_existence
        label_all = sample.label_ordered
        flag = sample.object_based.flag
        label_task = sample.object_based.label_task
        id = sample.id
        label_existence,label_all,label_task,id = map(
                torch.tensor, (label_existence,label_all,label_task,id))
        label_existence=label_existence.float()
        adj_type , char = flag
        adj_type_ohe = torch.nn.functional.one_hot(torch.tensor(adj_type), self.ndirections)
        char_ohe = torch.nn.functional.one_hot(torch.tensor(char), self.nclasses_existence)
        flag = torch.cat((adj_type_ohe,char_ohe),dim=0)
        flag = flag.float()
        label_task = label_task.view((-1))
        return img,seg,label_existence,label_all,label_task,id, flag

def inputs_to_struct(inputs):
    """
    inputs_to_struct _summary_

    Args:
        inputs (_type_): _description_

    Returns:
        _type_: _description_
    """    
    img,label_existence,label_all,label_task,id, flag = inputs
    sample = SimpleNamespace()
    sample.image = img
    sample.label_occurence = label_existence
    sample.label_existence = label_existence
    sample.label_all = label_all
    sample.label_task = label_task
    sample.id = id
    sample.flag = flag
    return sample

class EMNISTAdjDatasetNew2(EMNISTAdjDatasetBase):
    """
    EMNISTAdjDatasetNew2 _summary_

    Args:
        EMNISTAdjDatasetBase (_type_): _description_
    """    
    def __init__(self, root, nclasses_existence, ndirections, nexamples=None, split=False, mean_image=None,
                 direction=0):
        super(EMNISTAdjDatasetNew2, self).__init__(root, nclasses_existence, ndirections, nexamples, split, mean_image)
        self.direction = direction

    def __getitem__(self, index):
        obj_per_row = 5
        obj_per_col = 1
        nclasses_existence = 47
        edge_class = nclasses_existence

        root = self.get_root_by_index(index)
        fname = os.path.join(root, '%d_img.jpg' % index)
        img = Image.open(fname).convert('RGB')
        img = transforms.ToTensor()(img)
        img = 255 * img
        if self.mean_image is not None:
            img -= self.mean_image
            img = img.float()
        sample = self.get_raw_sample(index)
        label_existence = sample.label_existence
        label_all = sample.label_ordered
        flag = sample.flag
        # label_task = sample.label_task
        # TODO understand this part
        id = sample.id
        flag[0] = self.direction
        adj_type, char = flag
        r, c = (label_all == char).nonzero()
        r = r[0]
        c = c[0]
        # find the adjacent char
        if adj_type == 0:
            # right
            if c == (obj_per_row - 1):
                label_task = edge_class
            else:
                label_task = label_all[r, c + 1] # see line 129 in `itsik/code/v26/avatar_dataset.py` file
        if adj_type == 1:
            # left
            if c == 0:
                label_task = edge_class
            else:
                label_task = label_all[r, c - 1]
        if adj_type == 2:
            # Up
            if r==0:
                label_task = edge_class
                self.UP=self.UP+1
            else:
                label_task = label_all[r-1, c]
        if adj_type == 3:
            #Down
            if r==obj_per_col-1:
                label_task = edge_class
            else:
                label_task = label_all[r+1, c ]

        label_existence, label_all, label_task, id = map(
            torch.tensor, (label_existence, label_all, label_task, id))
        label_existence = label_existence.float()
        adj_type, char = flag
        adj_type_ohe = torch.nn.functional.one_hot(torch.tensor(adj_type), self.ndirections)
        char_ohe = torch.nn.functional.one_hot(torch.tensor(char), self.nclasses_existence)
        flag = torch.cat((adj_type_ohe, char_ohe), dim=0)
        flag = flag.float()
        label_task = label_task.view((-1))
        return img, label_existence, label_all, label_task, id, flag
        # TODO edit the label task properly
