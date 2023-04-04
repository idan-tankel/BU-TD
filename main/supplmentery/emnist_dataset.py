import os
import pickle
from types import SimpleNamespace

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
        self.splitsize = 1000
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
    """MARKED FOR DELETION
    TODO make the class and the one coming after it 1 class
    TODO add conv2d to compute the adjecency matrix as done in the new versions in beta branch
    """
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
    ***DEPRECATED***
    This method will passed over to use the `inputs` class __init__ and will removed in the next revision
    inputs_to_struct Structure the inputs from a list to more complex object

    Args:
        inputs (`List[torch.Tensor]`): Long list of tensors representing all the inputs - image,label_existence,label_all,label_task,id,flag

    Returns:
        _type_: _description_
    """    
    sample = SimpleNamespace(image = inputs[0], label_existence = inputs[1], label_all = inputs[2], label_task = inputs[3], id = inputs[4], flag = inputs[5])
    # compute the label for each direction around each sample
    # each sample has shape of (number_of_rows,number_of_columns) which is in the config file
    # we will use a trick - we will pad the existing tensor (label_all) and then
    # instead of acting on each element a function that will return the 4 surrounding elements, we will use a convolution with the following filter
    # 0 1 0
    # 1 0 1
    # 0 1 0
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
        obj_per_row = 6 # TODO edit this value to be the same as in the config file of create_dataset.py since this it cause exception
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
        adj_type, char = flag
        r, c = (label_all == char).nonzero()
        r = r[0]
        c = c[0]
        # find the adjacent char
        # TODO use conv2d for that! much faster
        if adj_type == 0:
            # right
            # TODO when trying to find the right element of the 
            if c == (obj_per_row - 1):
                label_task = edge_class
            else:
                label_task = label_all[r, c + 1] # see line 129 in `itsik/code/v26/avatar_dataset.py` file
        elif adj_type == 1:
            # left
            if c == 0:
                label_task = edge_class
            else:
                label_task = label_all[r, c - 1]
        elif adj_type == 2:
            # Up
            if r==0:
                label_task = edge_class
                self.UP=self.UP+1
            else:
                label_task = label_all[r-1, c]
        elif adj_type == 3:
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
