from PIL import Image
import pickle
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from types import SimpleNamespace
import numpy as np
from .funcs import AutoSimpleNamespace

class BisectionVernierDatasetBase(data.Dataset):
    def __init__(self, root, nexamples = None, split = False,mean_image = None):
        self.root = root
        self.split = split
        self.splitsize=1000
        self.mean_image = mean_image
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

    def get_raw_sample(self, index):
        root = self.get_root_by_index(index)
        data_fname=os.path.join(root,'%d_raw.pkl' % index)
        with open(data_fname, "rb") as new_data_file:
            raw_sample = pickle.load(new_data_file)
        return raw_sample

    def __len__(self):
        return self.nexamples

class BisectionVernierDataset(BisectionVernierDatasetBase):
    def __getitem__(self, index):
        root = self.get_root_by_index(index)
        fname=os.path.join(root,'%d_img.jpg' % index)
        img = Image.open(fname).convert('RGB')
        img = transforms.ToTensor()(img)
        img = 255*img
        if self.mean_image is not None:
            img -= self.mean_image
            img=img.float()
        sample = self.get_raw_sample(index)
        flag = sample.flag
        label_flag = sample.label_flag
        label_all = sample.label_all.astype(np.int)
        id = sample.id
        # from IPython.core.debugger import Pdb; ipdb = Pdb(); ipdb.set_trace()
        keypoints = np.array(sample.keypoints)[:,:2]
        
        label_all,label_flag,id,keypoints = map(
                torch.tensor, (label_all,label_flag,id,keypoints))
        flag = torch.nn.functional.one_hot(torch.tensor(flag), 2)
        flag = flag.float()
        label_task = label_flag
        label_task = label_task.view((-1))
        return img,label_all,label_flag,label_task, id, flag,keypoints


def inputs_to_struct_basic(inputs):
    image,label_all,label_flag,label_task, id, flag,keypoints = inputs
    sample = AutoSimpleNamespace(locals(), image,label_all,label_flag,label_task, id, flag,keypoints).tons()
    return sample



