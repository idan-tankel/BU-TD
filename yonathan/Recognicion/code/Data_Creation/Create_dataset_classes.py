import random
from enum import Enum

import skimage
import torch
from torch.utils.data import Dataset
from imgaug import augmenters as iaa

import argparse
import os
from pathlib import Path

import numpy as np
import skimage.io
import torchvision
from torchvision import datasets
from torchvision import transforms

try:
    from utils import Download_raw_omniglot_data
except ModuleNotFoundError:
    pass

from typing import Union


# Here we define all our classes like transforms, data set type, Sample, Char.


class DsType(Enum):
    """
    Here we define our data set types.
    """
    Emnist = 'Emnist'
    Omniglot = 'Omniglot'
    Fashionmnist = 'Fashionmnist'

    def __str__(self):
        return self.value


class General_raw_data(Dataset):
    """
    We created data class to handle retrieving raw character images.
    Support initialization, merging(for Omniglot) and getitem.
    """

    def __init__(self, download_dir: str):
        """

        Args:
            download_dir: The path download the raw data into.
        """
        self.download_dir = download_dir  # Download path.
        self.raw_images = None  # The raw images.
        self.labels = None  # The labels.
        self.nclasses = None  # The number of classes.
        self.shape = (1, 28, 28)  # The image shape.

    def Merge_two_datasets(self, train_raw_dataset: Dataset, test_raw_dataset: Dataset) -> \
            tuple[list[np.ndarray], list[np.int]]:
        """
        We merge two datasets into one list.
        Args:
            train_raw_dataset: The train dataset.
            test_raw_dataset:  The test dataset.

        Returns: The merged raw images, labels.

        """
        # Initialize the raw images list.
        Images_arranged = [[] for _ in range(self.nclasses)]
        # For each dataset iterate over images, labels into the list.
        for data_set in [train_raw_dataset, test_raw_dataset]:
            for (img, label) in data_set:
                Images_arranged[label].append(np.array(img).reshape(self.shape))
        # Make all images in one path.
        Images_arranged = sum(Images_arranged, [])
        # The number of images per single label.
        num_images_per_label = len(Images_arranged) // self.nclasses
        # Make all labels in one path.
        labels_arranged = sum([[k] * num_images_per_label for k in range(self.nclasses)], [])
        return Images_arranged, labels_arranged

    def __getitem__(self, index: int) -> tuple:
        """
        Args:
            index: The image index.

        Returns: The image, label in that index.
        """
        return self.raw_images[index], self.labels[index]

    def __len__(self):
        return len(self.raw_images)


class Emnist_raw_data(General_raw_data):
    """
    The Emnist data-set.
    """

    def __init__(self, download_dir):
        """
        Args:
            download_dir: The path download the raw data into.
        """
        super().__init__(download_dir)
        # Rotation transform, as the emnist images come rotated and flipped.
        self.emnist_transform = torchvision.transforms.Compose([
            lambda data_image: torchvision.transforms.functional.rotate(
                data_image, -90),
            lambda data_image: torchvision.transforms.functional.hflip(data_image),
            torchvision.transforms.ToTensor()
        ])
        # Train Data.
        train_raw_dataset = datasets.EMNIST(
            download_dir,
            download=True,
            split='balanced',
            train=True,
            transform=self.emnist_transform)

        # Test Data.
        test_raw_dataset = datasets.EMNIST(
            download_dir,
            download=True,
            split='balanced',
            train=False,
            transform=self.emnist_transform)

        self.nclasses = 47  # There are 47 classes.
        self.raw_images, self.labels = self.Merge_two_datasets(train_raw_dataset,
                                                               test_raw_dataset)  # Merge the two datasets.
        self.num_examples_per_character = len(self.raw_images) // 47


class FashionMnist_raw_data(General_raw_data):
    """
    The Fashionmnist data-set.
    """

    def __init__(self, download_dir: str):
        """
        Fashionmnist raw Data.
        Args:
            download_dir: The path download the raw Data into.
        """
        super().__init__(download_dir)
        # Train Data.
        train_raw_dataset = datasets.FashionMNIST(
            download_dir, download=True)
        # Test Data.
        test_raw_dataset = datasets.FashionMNIST(
            download_dir, train=False, download=True)
        self.nclasses = 10  # Ten classes.
        # Merge the data.
        self.raw_images, self.labels = self.Merge_two_datasets(train_raw_dataset, test_raw_dataset)
        self.num_examples_per_character = len(self.raw_images) // 10


class Omniglot_data_set(General_raw_data):
    """
    The Omniglot data-set.
    """

    def __init__(self, download_dir, language_list):
        """
        Args:
            download_dir: The path download the raw Data into.
            language_list: The list of desired languages.
        """
        super().__init__(download_dir=download_dir)
        # The raw data path.
        self.raw_data_source = os.path.join(Path(__file__).parents[3], 'data/Omniglot/RAW/omniglot-py/Unified')
        # Make the Omniglot dictionary.
        Omniglot_dict = Download_raw_omniglot_data(download_dir)
        # List of all languages, ordered by number of characters.
        All_languages = list(Omniglot_dict)
        # Transform to make tensors.
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.functional.invert, transforms.Resize(self.shape[1:])])
        self.raw_images = []
        # Iterate over all languages in language_list.
        for lan_idx in language_list:
            lan = os.path.join(self.raw_data_source, All_languages[lan_idx])  # The langauge
            language_images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(lan) for f in
                               filenames]  # All character images.
            language_images.sort()  # Sort to be ordered.
            language_images = [transform(skimage.io.imread(im_file)) for im_file in language_images]  # Make image.
            self.raw_images.extend(language_images)  # Add all language characters images.
        self.num_examples_per_character = 20  # 20 images per character.
        self.nclasses = len(self.raw_images) // 20  # The number of labels.
        self.labels = sum([self.num_examples_per_character * [i] for i in range(self.nclasses)], [])  # Merge into one
        # list.


class GenericDatasetParams:
    """
    The Generic data-set parameters clas.
    Supports needed things for creating like desired classes, transformations parameters.
    """

    def __init__(self, ds_type: DsType, num_cols: int, num_rows: int):
        """
        Here we define Generic Data set params for the transformation.
        We declare on variables, any user will need to define.
        Args:
            ds_type: The data-set type.
            num_cols: The number of columns.
            num_rows: The number of rows
        """
        self.ds_type: DsType = ds_type  # The data-set type.
        self.letter_size: int = 28  # The letter size is always 28.
        self.min_scale = None  # The minimal scale factor we apply.
        self.max_scale = None  # The maximal scale factor we apply.
        self.ngenerate = None  # The number of samples to generate per chosen sequence.
        self.min_shift = None  # The minimal shift.
        self.max_shift = None  # The maximal shift.
        self.create_CG_test: bool = False  # Whether to create the CG test.
        # Only For emnist, we use partial part of the classes.
        self.use_only_valid_classes: bool = ds_type is DsType.Emnist
        self.image_size = None  # The image size.
        self.nsamples_train = None  # The number of train samples.
        self.nsamples_test = None  # The number of test samples.
        self.nsamples_val = None  # The number of samples for the CG test.
        self.Data_path = os.path.join(Path(__file__).parents[2],
                                      f'data/{str(ds_type)}/RAW')  # The path to the raw Data.
        # This is the rule, we define the number of samples we generate per chosen sequence.
        if num_rows == 1 or num_cols == 1:
            self.ngenerate = num_rows - 1 if num_cols == 1 else num_cols - 1
        else:
            self.ngenerate = (num_rows - 1) * (num_cols - 1)  # This rule worked the best for us.


class EmnistParams(GenericDatasetParams):
    """
    The Emnist specific parameters.
    """

    def __init__(self, ds_type: DsType, num_cols: int, num_rows: int):
        """
        Here we define the Emnist data-set specification.
        Those params help the model to generalize and the output images that are understandable.
        Args:
            num_cols: The number of columns.
            num_rows: The number of rows
        """
        super(EmnistParams, self).__init__(ds_type=ds_type, num_cols=num_cols, num_rows=num_rows)
        self.min_scale = 1.0
        self.max_scale = 1.1
        self.min_shift = 2.0
        self.max_shift = 0.2 * 28
        self.generalize = True
        self.create_CG_test = True
        self.image_size = [130, 200]
        self.ngenerate = 5
        self.nsamples_train = 20000
        self.nsamples_test = 2000
        self.nsamples_val = 2000
        self.raw_data_set = Emnist_raw_data(self.Data_path)


class FashionMnistParams(GenericDatasetParams):
    """
    The Fashinmnist specification parameters.
    """

    def __init__(self, ds_type, num_cols: int, num_rows: int):
        """
        Here we define the Fashion-Mnist data-set specification.
        Those params help the model to generalize and the output images are understandable.
        Args:
            num_cols: The number of columns.
            num_rows: The number of rows
        """
        super(FashionMnistParams, self).__init__(ds_type=ds_type, num_cols=num_cols, num_rows=num_rows)
        self.min_scale = 1.2
        self.max_scale = 1.5
        self.min_shift = 2.0
        self.max_shift = 0.2 * 28
        self.generalize = False
        self.image_size = [112, 130]
        self.nsamples_train = 25000
        self.nsamples_test = 2000
        self.nsamples_val = 2000
        self.raw_data_set = FashionMnist_raw_data(self.Data_path)


class OmniglotParams(GenericDatasetParams):
    """
    Here we define the Omniglot data-set specification.
    """

    def __init__(self, ds_type: DsType, num_cols: int, num_rows: int, language_list):
        """
        Args:
            num_cols: The number of columns.
            num_rows: The number of rows
        """
        super(OmniglotParams, self).__init__(ds_type=ds_type, num_cols=num_cols, num_rows=num_rows)
        self.min_scale = 1.2
        self.max_scale = 1.5
        self.min_shift = 2.0
        self.max_shift = 0.2 * 28
        self.generalize = False
        self.image_size = [55, 200]
        self.nsamples_train = 20000
        self.nsamples_test = 2000
        self.nsamples_val = 2000
        self.raw_data_set = Omniglot_data_set(download_dir=self.Data_path, language_list=language_list)


class UnifiedDataSetType:
    """
    This class contains all data-set objects, according to the data-set type.
    """

    def __init__(self, ds_type: DsType, num_cols: int, num_rows: int, language_list: Union[list, None]):
        """
        Supports all datasets.
        Given dataset type we create the desired data-set specification.
        Args:
            ds_type: The data set types.
            num_cols: The number of cols
            num_rows: The number of rows.
        """

        if ds_type is DsType.Emnist:
            self.ds_obj = EmnistParams(ds_type=ds_type, num_cols=num_cols, num_rows=num_rows)
        if ds_type is DsType.Fashionmnist:
            self.ds_obj = FashionMnistParams(ds_type=ds_type, num_cols=num_cols, num_rows=num_rows)
        if ds_type is DsType.Omniglot:
            self.ds_obj = OmniglotParams(ds_type=ds_type, num_cols=num_cols, num_rows=num_rows,
                                         language_list=language_list)


class CharInfo:
    """
    The character information class.
    contains the image, scale, shift, coordinates.
    """

    def __init__(self, parser: argparse, raw_dataset, prng: random, sample_id: int, sample_chars: list):
        """
        Args:
            parser: The option parser.
            prng: The random generator.
            sample_id: The sample id.
            sample_chars: The sampled characters list
        """
        min_scale = parser.min_scale
        max_scale = parser.max_scale
        min_shift = parser.min_shift
        max_shift = parser.max_shift
        self.scale = prng.rand() * (max_scale - min_scale) + min_scale
        letter_size = parser.letter_size
        new_size = int(self.scale * parser.letter_size)
        shift = prng.rand(2) * (max_shift - min_shift) + min_shift
        y, x = shift.astype(np.int)
        start_row, start_col = np.unravel_index(sample_id, [parser.num_rows, parser.num_cols])
        c = start_col + 0.5  # start in the middle of the column.
        r = start_row + 0.1  # Start in the 0.1 of the row.
        image_height, image_width = parser.image_size  # The desired image dimension.
        stx = int(c * parser.letter_size + x)
        stx = max(0, stx)  # Ensure it's non-negative.
        stx = min(stx, image_width - new_size)
        sty = int(r * parser.letter_size + y)
        sty = max(0, sty)
        sty = min(sty, image_height - new_size)
        num_examples_per_character = raw_dataset.num_examples_per_character
        self.label = sample_chars[sample_id]
        self.label_id = prng.randint(0, num_examples_per_character)  # Choose a specific character image.
        # The index in the data-loader.
        self.img_id = num_examples_per_character * self.label + self.label_id
        self.img, _ = raw_dataset[self.img_id]  # The character image.
        self.letter_size = letter_size
        self.location_x = stx
        self.nclasses = raw_dataset.nclasses
        self.location_y = sty
        self.edge_to_the_right = start_col == parser.num_cols - 1 or sample_id == parser.num_characters_per_sample - 1
        # Scaling character and getting the desired location to plant the character.
        h, w = new_size, new_size
        sz = (1, h, w)
        self.img = skimage.transform.resize(self.img, sz, mode='constant')  # Apply the resize transform.
        self.stx = self.location_x
        self.end_x = stx + w
        self.sty = self.location_y
        self.end_y = sty + h


class Sample:
    """
    Class containing all characters information including the Characters themselves, the sampled sequence,
    the query part id.
    """

    def __init__(self, parser, query_part_id: int, adj_type: int, chars: list[CharInfo], sample_id: int):
        """

        Args:
            parser: The parser.
            query_part_id: The index we query about.
            adj_type: The direction we query about.
            chars: The list of all characters in the sample.
        """
        #  self.sampled_chars = sampled_chars
        self.query_part_id = query_part_id  # The index we query about.
        self.direction_query = adj_type  # The direction I query about.
        self.chars = chars  # All character objects.
        self.query_coord = np.unravel_index(query_part_id,
                                            [parser.num_rows, parser.num_cols])  # Getting the place we query about.

        self.image = np.zeros((1, *parser.image_size),
                              dtype=np.float32)  # Initialize with zeros, will be updated in create_image_matrix.
        self.label_existence = Get_label_existence(chars, chars[0].nclasses)  # The label existence.
        self.label_ordered = Get_label_ordered(chars)  # The label ordered.
        self.sample_id = sample_id  # The sample id.


class DataAugmentClass:
    """
    class performing the image augmentation.
    """

    def __init__(self, seed: int):
        """
        Args:
            seed: The seed for the generation.
        """

        color_add_range = int(0.2 * 255)
        rotate_deg = 15  # Rotate up to 16 degrees.
        self.aug = iaa.Sequential([iaa.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.1, 0.1)},
                                              rotate=(-rotate_deg, rotate_deg), mode='edge', name='affine'), ],
                                  random_state=0)
        self.aug.append(iaa.Add((-color_add_range, color_add_range)))  # only for the image not the segmentation
        self.aug_seed = seed  # The random seed.

    def __call__(self, image: np.ndarray):
        """
        Applying the data augmentation on the image.
        Args:
            image: The image we want to apply the transform on.

        Returns: The augmented image.

        """
        aug = self.aug
        aug_seed = self.aug_seed
        aug.seed_(aug_seed)
        aug_images = aug.augment_images([image])
        return aug_images[0]


# TODO - CHANGE THE CLASS.
class MetaData:
    """
     Saves the parser, and the number of samples per dataset.
    """

    def __init__(self, parser: argparse, nsamples_per_data_type_dict: dict):
        """

        Args:
            parser: The parser.
            nsamples_per_data_type_dict: The nsamples dictionary.
        """

        self.nsamples_dict = nsamples_per_data_type_dict
        self.parser = parser


def Get_label_existence(chars: list[CharInfo], nclasses: int) -> torch:
    """
    Generating the label_existence label.
    Args:
        chars: The characters list.
        nclasses: The number of classes.

    Returns: The label_existence flag, telling for each entry whether the character is in the image.
    """
    label_existence = torch.zeros(nclasses)  # Initially all zeros.
    for char in chars:  # for each character in the image.
        label_existence[char.label] = 1.0  # Set 1 in the label.
    return label_existence


def Get_label_ordered(chars: list[CharInfo]) -> torch:
    """
    Generate the label_ordered label.
    Args:
        chars: The characters list.

    Returns: list of all characters arranged in order.
    """
    row = list()
    label_ordered = []  # All the rows.
    for char in chars:  # Iterate for each character in the image.
        character = char.label
        row.append(character)  # All the characters in the row.
        if char.edge_to_the_right:  # The row ended, so we add it to the rows list.
            label_ordered.append(row)
            row = list()

    label_ordered = torch.tensor(label_ordered)
    return label_ordered
