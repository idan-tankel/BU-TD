import os
import torch
import pickle
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as T


class cifar_dataset(data.Dataset):
    def __init__(self,root,nclasses:int = 10,split:bool= False,splitsize:int = 1000,ds_type:str="train"):
        self.root = root
        self.nclasses = nclasses
        self.split = split
        self.splitsize = splitsize
        self.ds_type = ds_type
        self.is_train = ds_type == 'train'
        self.transform = T.Compose([T.ToTensor(), T.Normalize( mean = (0.4914, 0.4822, 0.4465), std = (0.247, 0.243, 0.261)) ])
        
    def __getitem__(self, index):
        path = os.path.join(self.root,self.ds_type)
        fname = os.path.join(path, '%d_img.jpg' % index)
        # Opening the image and converting to Tensor
        img = Image.open(fname).convert('RGB')
       # img = transforms.ToTensor()(img)
       # img =T.ToTensor()(img)
        img = self.transform(img)
        pkl_file = os.path.join(path, '%d.pkl' % index)
        with open(pkl_file,'rb') as f :
         label = pickle.load(f)
        return img,label

    def __len__(self):
        if self.is_train:
          return 50000
        else:
         return 10000

def inputs_to_struct():
    pass

