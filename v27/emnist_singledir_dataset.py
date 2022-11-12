from PIL import Image
import pickle
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from types import SimpleNamespace
import numpy as np
import random # torch random transforms uses random
from v27.emnist_dataset import EMNISTAdjDatasetBase, inputs_to_struct_basic as inputs_to_struct
from v27.dataset_storage import Cache


class EMNISTAdjDatasetNew2(EMNISTAdjDatasetBase):
    def __init__(self, root, nclasses_existence,ndirections, nexamples = None, split = False,mean_image = None, right = True):
        super(EMNISTAdjDatasetNew2, self).__init__(root, nclasses_existence,ndirections, nexamples, split,mean_image)
        self.right = right

    def __getitem__(self, index):
        sample = self.get_base_raw_sample(index)
        return self.process_sample(sample)

    def process_sample(self,sample):
        img,seg = sample.img, sample.seg
        obj_per_row = 6
        nclasses_existence = 47
        edge_class = nclasses_existence
        img,seg = sample.img, sample.seg
        img,seg = map(
                transforms.ToTensor(), (img,seg))
        img,seg = 255*img,255*seg
        if self.mean_image is not None:
            img -= self.mean_image
            img=img.float()
        
        label_existence = sample.label_existence
        label_all = sample.label_ordered
        flag = sample.flag
        # label_task = sample.label_task
        id = sample.id
        
        if self.right:
            flag[0]=0
        else:
            flag[0]=1
        adj_type , char = flag
        r,c = (label_all== char).nonzero()
        r=r[0]
        c=c[0]
        # find the adjacent char
        if adj_type == 0:
            # right
            if c == (obj_per_row - 1):
                label_task = edge_class
            else:
                label_task = label_all[r, c + 1]
        else:
            # left
            if c == 0:
                label_task = edge_class
            else:
                label_task = label_all[r, c - 1]
        
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
