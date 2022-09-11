import os
import pickle

import numpy as np
import torch
import torch.utils.data as data
import torchvision
from skimage import io
from torchvision import transforms

def folder_size(path: str) -> int:
    """
    :param path: path to the raw data.
    :returns The size of a given folder.
    """
    size = 0
    for _ in os.scandir(path):
        size += 1
    return size

def create_dict(path: str) -> dict:
    """
    :param path: Path to the data.
    :return: Dictionary assigning for each language its number of characters.
    """
    dict_language = {}
    cnt = 0
    for ele in os.scandir(path):
        path_new = ele
        dict_language[cnt] = folder_size(path_new)
        cnt += 1
    return dict_language

def get_raw_data(download_dir:str, dataset:str)->tuple:
    """
    Args:
        download_dir: The directory to download the raw data into.
        dataset: The dataset type e.g. emnist, cifar10.

    Returns: The images, labels, the raw character size, the number of channels.
    """
    # Getting the raw train, test data.
    train_raw_data,test_raw_data, rotate = None,None,None
    nchannels = 1
    if dataset == 'emnist': # The balanced emnist dataset.
        train_raw_data = torchvision.datasets.EMNIST(download_dir, split='balanced', train=True, download=True)
        test_raw_data = torchvision.datasets.EMNIST(download_dir, split='balanced', train=False, download=True)
        rotate = (1,0)
        nchannels = 1
        shape = (1,28,28)
    if dataset == 'cifar10': # The cifar10 dataset.
        train_raw_data = torchvision.datasets.CIFAR10(download_dir, train=True, download=True)
        test_raw_data = torchvision.datasets.CIFAR10(download_dir, train=False, download=True)
        rotate = (2,0,1)
        nchannels = 3
    if dataset == 'cifar100': # The cifar100 dataset.
        train_raw_data = torchvision.datasets.CIFAR100(download_dir, train=True, download=True)
        test_raw_data = torchvision.datasets.CIFAR100(download_dir, train=False, download=True)
        rotate = (2,0,1)
        nchannels = 3
    if dataset == 'FashionMnist': # The fashion mnist dataset.
        train_raw_data = torchvision.datasets.FashionMNIST(download_dir, train=True, download=True)
        test_raw_data = torchvision.datasets.FashionMNIST(download_dir, train=False, download=True)
        rotate = (0,1)
        nchannels = 1
        shape = (1,28,28)
    # From dataset to list.
    images = []
    labels = []
    for raw_data in [train_raw_data, test_raw_data]:
        for img, lbl in raw_data:
            labels.append(lbl)
            img = np.array(img)
            images.append(img)
    letter_size = images[0].shape[1]
    num_labels = len(set(labels))
    images_arranged = [[] for _ in range(num_labels)]
    labels_arranged = [[] for _ in range(num_labels)]
    # Arranging the data according to the labels.
    for idx in range(len(labels)):
        index = labels[idx]
        img = images[idx].transpose(rotate)
        if dataset == 'emnist' or dataset == 'FashionMnist':
         img = img.reshape(shape)
        images_arranged[index].append(img)
        labels_arranged[index].append(index)
    # From list of lists to list.
    images_arranged = sum(images_arranged, [])
    labels_arranged = sum(labels_arranged, [])
    return images_arranged, labels_arranged, letter_size,nchannels

class DataSet(data.Dataset):
    def __init__(self, data_dir:str, dataset:str,language_list:list,raw_data_source:str):
        """
        Args:
            data_dir: The data we want to store the raw data into, in order to not re-download the raw data again.
            dataset: The dataset type.
            language_list: The language list for the Omniglot dataset.
            raw_data_source: # The Raw data source for the Omniglot dataset.
        """
        assert dataset in ['emnist', 'cifar10', 'cifar100', 'FashionMnist', 'omniglot']
        data_dir = os.path.join(data_dir,'RAW')
        download_raw_data_dir = os.path.join(data_dir, f'{dataset}_raw', ) # The path we download the raw data into.
        if dataset != 'omniglot':
         self.images, self.labels, self.letter_size, self.nchannels = get_raw_data(download_raw_data_dir, dataset = dataset)
        else:
         self.images, self.labels, self.letter_size, self.nchannels = get_raw_data_omniglot(language_list,raw_data_source )
        self.nclasses = len(set(self.labels))
        self.num_examples_per_character = len(self.labels) // self.nclasses

    def __getitem__(self, index:int)->tuple:
        """
        Args:
            index: The image index.

        Returns: The image, label in that index.
        """
        image = self.images[index]
        label = self.labels[index]
        image = image
        return image, label

    def __len__(self):
        return len(self.images)

def get_raw_data_omniglot(languages_list:list,raw_data_source:str)->tuple:
    """
    Args:
        languages_list: The language list.
        raw_data_source: The source for the raw data.

    Returns: The raw images, labels, letter and the number of channels.

    """
    images = []  # The images list.
    labels = []  # The labels list.
    sum, num_charcters = 0,0
    transform = transforms.Compose([transforms.ToTensor(), transforms.functional.invert,transforms.Resize([28, 28])])  # Transforming to tensor + resizing.
    dictionary = create_dict(raw_data_source)  # Getting the dictionary.
    language_list = os.listdir(raw_data_source)  # The languages list.
    for language_idx in languages_list:  # Iterating over all the list of languages.
        lan = os.path.join(raw_data_source, language_list[language_idx])
        for idx, char in enumerate(os.listdir(lan)):
            char_path = os.path.join(lan, char)
            for idx2, sample in enumerate(os.listdir(char_path)):
                sample_path = os.path.join(char_path, sample)
                img = io.imread(sample_path)
                idx_torch = idx + sum
                img = transform(img)
                images.append(img)
                labels.append(idx_torch)
            num_charcters = num_charcters + 1
        sum = sum + dictionary[language_idx]
    images = torch.stack(images, dim=0)
    letter_size = images[0].shape[1]

    return images, labels, letter_size, 1

    # if True, generate multiple examples (image,label) pairs from each image, else generate a single example
    # generate multiple examples (image,label) pairs from each image
    # We don't exclude (char,border) or (border,char) pairs as this drastically limits the available valid training examples
    # Therefore, do not query near borders
# valid_queries = np.arange(sample_nchars)
#  near_border = np.arange(0, sample_nchars, obj_per_row)
# valid_queries_right = np.setdiff1d(valid_queries, near_border)
# valid_queries_left = np.setdiff1d(valid_queries, near_border)
#  dictionary = create_dict(parser.path_data_raw)  # The dictionary assigning for each language its number of characters.
#  dictionary = dict(sorted(dictionary.items(), key=lambda item: item[1]))