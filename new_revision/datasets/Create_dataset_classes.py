import argparse
import random
from enum import Enum, auto

import numpy as np
from imgaug import augmenters as iaa


class DsType(Enum):
    Emnist = auto()
    Kmnist = auto()
    Omniglot = auto()
    Fashionmist = auto()

    def from_enum_to_str(self) -> str:
        if self == DsType.Emnist:
            return "emnist"
        if self == DsType.Kmnist:
            return "kmnist"
        if self == DsType. Omniglot:
            return "omniglot"
        if self == DsType.Fashionmist:
            return "fashionmnist"
        else:
            print(f"Dataset {type(self)} Is not Supported by DsType class")
            raise ValueError
            


class DatasetParams:
    def __init__(self, ds_type: DsType):
        """
        Args:
            flag_at: The model flag.
        """
        self.ds_type = ds_type
        self.minscale = None
        self.maxscale = None
        self.generelize = None
        self.use_only_valid_classes = None


class EmnistParams:
    def __init__(self, ds_type: DsType):
        super(EmnistParams, self).__init__(ds_type)
        self.minscale = 0.9


class Sample:
    # Class containing all information about the sample, including the image, the flags, the label task.
    def __init__(self, image, sample_id, label_existence, label_ordered, query_part_id, label_task, flag, is_train):
        """
        #Class containing all information about the sample, including the image, the flags, the label task.
        Args:
            image: The image.
            sample_id: The id.
            label_existence: The label existence.
            label_ordered: The label ordered, containing all the characters arranged.
            query_part_id: The index we query about.
            label_task: The label task.
            flag: The flag.
            is_train: Whether the sample is part of the training set.
        """
        self.image = image
        self.id = sample_id
        self.label_existence = label_existence.astype(np.int)
        self.label_ordered = label_ordered
        self.query_part_id = query_part_id
        self.label_task = label_task
        self.flag = flag
        self.is_train = is_train


class ExampleClass:
    def __init__(self, sampled_chars: list, query_part_id: int, relation: int, chars: list):
        """
        ## deprecated
        Args:
            sampled_chars: The sampled characters
            query_part_id: The index we query about.
            adj_type: The relation / direction we query about.
            chars: The list of all characters in the sample.
        """
        self.sampled_chars = sampled_chars
        self.query_part_id = query_part_id
        self.relation = relation
        self.chars = chars


class DataAugmentClass:
    """
    class performing the augmentation.
    """

    def __init__(self, seed: int):
        """
        Args:
            seed: The seed for the generation.
        """

        color_add_range = int(0.15 * 255)
        rotate_deg = 10
        # translate by -20 to +20 percent (per axis))
        self.aug = iaa.Sequential([iaa.Affine(translate_percent={
                                  "x": (-0.03, 0.03), "y": (-0.1, 0.1)},  rotate=(-rotate_deg, rotate_deg), mode='edge', name='affine'), ], random_state=0)
        # only for the image not the segmentation
        self.aug.append(iaa.Add((-color_add_range, color_add_range)))
        self.aug_seed = seed

    def __call__(self, image):
        """
        Applying the data augmentation on the image.
        Args:
            image: 

        Returns:

        """
        aug = self.aug
        aug_seed = self.aug_seed
        aug.seed_(aug_seed)
        aug_images = aug.augment_images([image])
        return aug_images[0]


class CharInfo:
    def __init__(self, parser: argparse, prng: random, label_ids: list, samplei: int, sample_chars: int):
        """
        Args:
            parser: The options parser.
            prng: The random generator.
            label_ids: The label ids list.
            samplei: The sample id.
            sample_chars: The sampled characters list
        """
        minscale = parser.minscale
        maxscale = parser.maxscale
        minshift = parser.minshift
        maxshift = parser.maxshift
        # place the chars on the image
        scale = prng.rand() * (maxscale - minscale) + minscale
        letter_size = parser.letter_size
        new_size = int(scale * parser.letter_size)
        shift = prng.rand(2) * (maxshift - minshift) + minshift
        y, x = shift.astype(np.int)
        origr, origc = np.unravel_index(
            samplei, [parser.num_rows_in_the_image, parser.nchars_per_row])
        c = origc + 1  # start from column 1 instead of 0
        r = origr + 1
        _, imageh, imagew = parser.image_size
        stx = c * parser.letter_size + x
        stx = max(0, stx)
        stx = min(stx, imagew - new_size)
        sty = int(r * parser.letter_size + y)
        sty = max(0, sty)
        sty = min(sty, imageh - new_size)

        self.label_id = label_ids[samplei]
        self.label = sample_chars[samplei]
        self.letter_size = letter_size
        self.scale = scale
        self.location_x = stx
        self.location_y = sty
        self.edge_to_the_right = origc == parser.nchars_per_row - \
            1 or samplei == parser.num_characters_per_sample - 1


class MetaData:
    def __init__(self, parser, nsampes_per_data_type_dict):
        self.nsamples_dict = nsampes_per_data_type_dict
        self.parser = parser
