from torch.utils.data import Dataset
import json
from collections import defaultdict
import torch
from PIL import Image
# from spacy.lang.en import English
# import torch.nn.functional as F
# import h5py
# from gensim.models import KeyedVectors
from torchvision import transforms
import os
from collections import defaultdict


class imSituDatasetGood(Dataset):
    def __init__(self, verb_dict, json_file, is_train,transformation=None, image_store_location='./images_512/', inference=False):
        """
        __init__ The basic dataset class for SWiG Dataset

        Args:
            verb_dict (_type_): The dictionary of verbs and their corresponding index
            json_file (_type_): _description_
            is_train (bool): _description_
            image_store_location (str, optional): _description_. Defaults to './images_512/'.
            inference (bool, optional): _description_. Defaults to False.
        """
        super(Dataset, self).__init__()
        self.inference = inference
        self.json_file = json_file
        self.image_store_location = image_store_location
        if not self.inference:
            self.word_file = './SWiG_jsons/imsitu_space.json'
            self.verb_dict = verb_dict
            self.verb_dict_revert = {v: k for k, v in self.verb_dict.items()}
            self.training_data = []
            if is_train:
                self.transformation = self.init_train_tranforms()
            else:
                self.transformation = self.init_val_tranforms()
            self.load_json()
        else:
            self.transformation = self.init_val_tranforms()
            self.training_data = []
            self.load_inference()
        # not the most efficient solution - irrelevant definition of transformation
        if transformation is not None:
            self.transformation = transformation

    def load_inference(self):
        with open(self.json_file) as f:
            for line in f:
                line = line.split('\n')[0]
                self.training_data.append(line)

    def load_json(self, ):
        with open(self.json_file) as f:
            train = json.load(f)

        d = {}
        for image in train:
            verb = train[image]["verb"]
            frame_length = len(train[image]["frames"][0]) - 1
            target = torch.zeros(1788)

            for role in train[image]["frames"][0]:
                if verb + '_' + role not in d:
                    d[verb + '_' + role] = len(d)
                if int(train[image]["bb"][role][0]) != -1:
                    target[d[verb + '_' + role]] = 1

            verb_idx = self.verb_dict[verb]
            self.training_data.append({'verb': torch.Tensor([verb_idx]), 'image_features': image.split('.')[
                                      0], 'frame_length': frame_length, 'roles': target, 'named_verb': verb, 'all': train[image]})

    def init_train_tranforms(self):
        """initialized the transform used on the images in the train data"""
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transformation = transforms.Compose([
            transforms.RandomResizedCrop(256, scale=(0.08, 1.0)),
            transforms.ColorJitter(hue=.05, saturation=.05, brightness=0.05),
            transforms.RandomGrayscale(p=0.3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        return transformation

    def init_val_tranforms(self):
        """initialized the transform used on the images in the validation data"""
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transformation = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            normalize,
        ])
        return transformation

    def get_im_tensor(self, image_path):
        im = Image.open(image_path).convert('RGB')
        tensor = self.transformation(im)
        im.close()
        return tensor

    def __len__(self):
        # return 50
        return len(self.training_data)

    def __getitem__(self, idx):
        data = self.training_data[idx]

        if self.inference:
            im_tensor = self.get_im_tensor(data)
            return {'image': im_tensor, 'im_name': data}

        im_tensor = self.get_im_tensor(os.path.join(self.image_store_location, fr"{data['image_features']}.jpg"))
        # preprocess the image and the text
        # return {'roles': data['roles'], 'nouns': data['nouns'], 'verb': data['verb'], 'image_features': torch.Tensor(image_features), 'image': im_tensor, 'all': data['all']}
        return {'label_all': data['verb'], 'img': im_tensor, 'text': data['named_verb'], 'frame_length': data['frame_length'], 'roles': data['roles'], 'im_name':  data['image_features'] + '.jpg'}
        # image --> img
        # frame_lenght
        # verb --> relation --> label_all (fixed length of 16)


if __name__ == "__main__":
    imSituDatasetGood()
