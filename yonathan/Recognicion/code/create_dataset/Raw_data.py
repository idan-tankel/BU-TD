import os

import numpy as np
import skimage.io
import torch.utils.data as data
import torchvision
from torchvision import transforms


def Get_raw_data(download_dir:str, dataset:str,language_list, raw_data_source)->tuple:
    """
    Args:
        download_dir: The directory to download the raw data into.
        dataset: The dataset type e.g. emnist, fashiomnist.
        language_list: The language list for omniglot
        raw_data_source: The raw data source for omniglot.

    Returns: The images, labels, the raw character size, the number of channels.

    """
    # Getting the raw train, test data.

    if dataset.from_enum_to_str() == 'omniglot':
        letter_size = 28
        nchannels = 1
        transform = transforms.Compose( [transforms.ToTensor(), transforms.functional.invert, transforms.Resize([28, 28])])
        images_arranged = []
        for lan_idx in language_list:
            language_list = os.listdir(raw_data_source)
            lan = os.path.join(raw_data_source, language_list[lan_idx])
            language_images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(lan) for f in filenames]
            language_images.sort()
            language_images = [transform(skimage.io.imread(im_file)) for im_file in language_images]
            images_arranged.extend(language_images)
        num_images_per_label = 20
        num_labels = len(images_arranged) // 20
        labels_arranged = sum([num_images_per_label * [i] for i in range(num_labels)], [])

    else:
        nchannels = 1
        if dataset.from_enum_to_str() == 'emnist':
            train_raw_dataset = torchvision.datasets.EMNIST(
                download_dir,
                download=True,
                split='balanced',
                train=True,
                transform=torchvision.transforms.Compose([
                    lambda img: torchvision.transforms.functional.rotate(img, -90),
                    lambda img: torchvision.transforms.functional.hflip(img),
                    torchvision.transforms.ToTensor()
                ])
            )
            test_raw_dataset = torchvision.datasets.EMNIST(
                download_dir,
                download=True,
                split='balanced',
                train=False,
                transform=torchvision.transforms.Compose([
                    lambda img: torchvision.transforms.functional.rotate(img, -90),
                    lambda img: torchvision.transforms.functional.hflip(img),
                    torchvision.transforms.ToTensor()
                ])
            )

            shape = (1, 28, 28)

        elif dataset.from_enum_to_str() == 'fashionmnist':
            train_raw_dataset = torchvision.datasets.FashionMNIST(download_dir, train=True, download=True)
            test_raw_dataset = torchvision.datasets.FashionMNIST(download_dir, train=False, download=True)
            shape = (1,28,28)

        elif dataset.from_enum_to_str() == 'kmnist':
            train_raw_dataset = torchvision.datasets.KMNIST(download_dir, train=True, download=True)
            test_raw_dataset = torchvision.datasets.KMNIST(download_dir, train=False, download=True)

        images = [np.array(train_raw_dataset[i][0]).reshape(shape) for i in range(len(train_raw_dataset))]
        images.extend([np.array(test_raw_dataset[i][0]).reshape(shape) for i in range(len(test_raw_dataset))])
        labels = [train_raw_dataset[i][1] for i in range(len(train_raw_dataset))]
        labels.extend([test_raw_dataset[i][1] for i in range(len(test_raw_dataset))])
        letter_size = images[0].shape[1]
        num_labels = len(set(labels))
        num_images_per_label = len(images) // num_labels
        # Arranging the data according to the labels.
        images_arranged = sum([[images[i]  for i in range(len(images)) if labels[i] == k ] for k in range(num_labels)],[])
        labels_arranged = sum( [[k] * num_images_per_label for k in range(num_labels)],[])
    return images_arranged, labels_arranged, letter_size,nchannels

class DataSet(data.Dataset):
    def __init__(self,parser, data_dir:str, dataset:str,language_list:list,raw_data_source:str):
        """
        Args:
            parser: The data-set options.
            data_dir: The data we want to store the raw data into, in order to not re-download the raw data again.
            dataset: The dataset type.
            language_list: The language list for the Omniglot dataset.
            raw_data_source: # The Raw data source for the Omniglot dataset.
        """
        data_dir = os.path.join(data_dir,'RAW')
        download_raw_data_dir = os.path.join(data_dir, f'{dataset.from_enum_to_str()}_raw', ) # The path we download the raw data into.
        self.images, self.labels, self.letter_size, self.nchannels = Get_raw_data(download_raw_data_dir, dataset = dataset,language_list = language_list,raw_data_source = raw_data_source)
        self.nclasses = len(set(self.labels))
        self.num_examples_per_character = len(self.labels) // self.nclasses
        parser.image_size = (self.nchannels,*parser.image_size)

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