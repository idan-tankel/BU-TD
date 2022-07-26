import os
import pickle

import numpy as np
import torch
import torch.utils.data as data
import torchvision
from skimage import io
from torchvision import transforms


# %% EMNIST dataset

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


def convert_raw_data_pytorch(download_dir, data_fname, dataset_type='cifar'):
    emnist_download_dir = download_dir
    emnist_split_type = 'balanced'
    labels = [[] for _ in range(2)]
    imgs = [[] for _ in range(2)]
    for k, train in enumerate((True, False)):
        if dataset_type == 'emnist':
            raw_data = torchvision.datasets.EMNIST(emnist_download_dir, emnist_split_type, train=train, download=True)
        if dataset_type == 'cifar':
            raw_data = torchvision.datasets.CIFAR10(emnist_download_dir, train=train, download=True)
        for img, lbl in raw_data:
            labels[k].append(lbl)
            img = np.array(img)
            img = img.reshape(-1)
            imgs[k].append(img)
    images_train, images_test = map(np.array, imgs)
    labels_train, labels_test = map(np.array, labels)

    with open(data_fname, "wb") as new_data_file:
        pickle.dump((images_train, images_test, labels_train, labels_test), new_data_file)


def get_raw_data(download_dir, dataset='emnist'):
    labels = [[] for _ in range(2)]
    imgs = [[] for _ in range(2)]
    if dataset == 'emnist':
        train_raw_data = torchvision.datasets.EMNIST(download_dir, split='balanced', train=True, download=True)
        test_raw_data = torchvision.datasets.EMNIST(download_dir, split='balanced', train=False, download=True)
    if dataset == 'cifar10':
        raw_data = torchvision.datasets.CIFAR10(download_dir, train=train, download=True)
        train_raw_data = torchvision.datasets.EMNIST(download_dir, train=True, download=True)
        test_raw_data = torchvision.datasets.EMNIST(download_dir, train=False, download=True)
    images = []
    labels = []
    for raw_data in [train_raw_data, test_raw_data]:
        for img, lbl in raw_data:
            labels.append(lbl)
            img = np.array(img)
          #  img = img.reshape(-1)  # TODO UNDERSTAND WHAT IS -1.
            images.append(img)

    num_labels = len(set(labels))
    images_arranged = [[] for _ in range(num_labels)]
    labels_arranged = [[] for _ in range(num_labels)]
    # Arranging the data.
    for idx in range(len(labels)):
        index = labels[idx]
        img = images[idx].transpose((1,0))
        images_arranged[index].append(img)
        labels_arranged[index].append(index)

    images_arranged = sum(images_arranged, [])
    labels_arranged = sum(labels_arranged, [])
    LETTER_SIZE = images_arranged[0].shape[0]
    return (images_arranged, labels_arranged, LETTER_SIZE)
 #   images_train, images_test = map(np.array, imgs)
   # labels_train, labels_test = map(np.array, labels)


# load the raw EMNIST dataset. If it doesn't exist download and process it
def load_raw_emnist_data(data_fname):
    with open(data_fname, "rb") as new_data_file:
        images_train, images_test, labels_train, labels_test = pickle.load(new_data_file)
    xs = np.concatenate((images_train, images_test))
    ys = np.concatenate((labels_train, labels_test))

    LETTER_SIZE = 32
    images = xs.reshape(len(xs), LETTER_SIZE)
    images = images.transpose((0, 2, 1))
    labels = ys
    total_bins = 8
    IMAGE_SIZE = [LETTER_SIZE * 4, LETTER_SIZE * total_bins]

    return images, labels, IMAGE_SIZE


class DataSet(data.Dataset):
    def __init__(self, data_dir, dataset='emnist'):
        assert dataset in ['emnist', 'cifar', 'fashionmnist']
        download_raw_data_dir = os.path.join(data_dir, f'{dataset}_raw', )
        raw_data_fname = os.path.join(data_dir, f'{dataset}-pyt.pkl')
        if dataset is not 'omniglot':
         self.images, self.labels, self.LETTER_SIZE = get_raw_data(download_raw_data_dir, dataset='emnist')
        else:
         self.images, self.labels, self.LETTER_SIZE = get_raw_data_omniglot(download_raw_data_dir, dataset='emnist')
        # TODO CHECK WE DON'T REDOWNALOD THE RAW DATA.
        self.nclasses = len(set(self.labels))
        self.num_examples_per_character = len(self.labels) // self.nclasses

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        image = image
        return image, label

    def __len__(self):
        return len(self.images)


class OmniglotDataLoader(data.Dataset):
    """
    Return data loader for omniglot.
    """

    def __init__(self, languages: list, Raw_data_source='/home/sverkip/data/omniglot/data/omniglot_all_languages'):
        """
        :param languages: The languages we desire to load.
        :param data_path:
        """
        images = []  # The images list.
        labels = []  # The labels list.
        sum, num_charcters = 0.0, 0
        transform = transforms.Compose([transforms.ToTensor(), transforms.functional.invert,
                                        transforms.Resize([28, 28])])  # Transforming to tensor + resizing.
        Data_source = '/home/sverkip/data/omniglot/data/omniglot_all_languages'
        dictionary = create_dict(Raw_data_source)  # Getting the dictionary.
        language_list = os.listdir(Raw_data_source)  # The languages list.
        for language_idx in languages:  # Iterating over all the list of languages.
            lan = os.path.join(Data_source, language_list[language_idx])
            for idx, char in enumerate(os.listdir(lan)):
                char_path = os.path.join(lan, char)
                for idx2, sample in enumerate(os.listdir(char_path)):
                    sample_path = os.path.join(char_path, sample)
                    img = io.imread(sample_path)
                    idx_torch = torch.tensor([idx + sum])
                    img = transform(img)
                    images.append(img)
                    labels.append(idx_torch)
                num_charcters = num_charcters + 1
            sum = sum + dictionary[language_idx]
        images = torch.stack(images, dim=0).squeeze()
        labels = torch.stack(labels, dim=0)
        self.images = images
        self.labels = labels
        self.nclasses = num_charcters
        self.num_examples_per_character = len(labels) // num_charcters

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        image = image
        return image, label

    # if True, generate multiple examples (image,label) pairs from each image, else generate a single example
    # generate multiple examples (image,label) pairs from each image
    # We don't exclude (char,border) or (border,char) pairs as this drastically limits the available valid training examples
    # Therefore, do not query near borders
# valid_queries = np.arange(sample_nchars)
#  near_border = np.arange(0, sample_nchars, obj_per_row)
# valid_queries_right = np.setdiff1d(valid_queries, near_border)
# valid_queries_left = np.setdiff1d(valid_queries, near_border)
