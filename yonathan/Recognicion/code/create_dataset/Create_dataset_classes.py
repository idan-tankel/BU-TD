import numpy as np
import argparse
import random
from imgaug import augmenters as iaa
from enum import Enum, auto

class DsType(Enum):
    Emnist = auto()
    Kmnist = auto()
    Omniglot =auto()
    Fashionmist =auto()

    def from_enum_to_str(self):
        if self == DsType.Emnist:
            return "emnist"
        if self == DsType.Kmnist:
            return "kmnist"
        if self == DsType. Omniglot:
            return "omniglot"
        if self == DsType.Fashionmist:
            return "Fashionmist"

class Sample:
    #Class containing all information about the sample, including the image, the flags, the label task.
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
    def __init__(self,sampled_chars:list,query_part_id:int,adj_type:int,chars:list):
        """
        Args:
            sampled_chars: The sampled characters
            query_part_id: The index we query about.
            adj_type: The direction we query about.
            chars: The list of all characters in the sample.
        """
        self.sampled_chars = sampled_chars
        self.query_part_id = query_part_id
        self.adj_type = adj_type
        self.chars = chars

class DataAugmentClass:
    """
    class performing the augmentation.
    """
    def __init__(self, seed:int):
        """
        Args:
            seed: The seed for the generation.
        """

        color_add_range = int(0.2 * 255)
        rotate_deg = 10
        # translate by -20 to +20 percent (per axis))
        self.aug = iaa.Sequential([iaa.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.1, 0.1)},  rotate=(-rotate_deg, rotate_deg), mode='edge', name='affine'), ], random_state=0)
        self.aug.append(iaa.Add((-color_add_range, color_add_range)))  # only for the image not the segmentation
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
    def __init__(self,parser:argparse, prng:random,label_ids:list,samplei:int,sample_chars:int ):
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
        origr, origc = np.unravel_index(samplei, [parser.num_rows_in_the_image, parser.nchars_per_row])
        c = origc + 1  # start from column 1 instead of 0
        r = origr
        _ , imageh , imagew = parser.image_size
        stx = c * parser.letter_size + x
        stx = max(0, stx)
        stx = min(stx, imagew - new_size)
        sty = r * parser.letter_size + y
        sty = max(0, sty)
        sty = min(sty, imageh - new_size)

        self.label_id = label_ids[samplei]
        self.label = sample_chars[samplei]
        self.letter_size = letter_size
        self.scale = scale
        self.location_x = stx
        self.location_y = sty
        self.edge_to_the_right = origc == parser.nchars_per_row - 1 or samplei == parser.num_characters_per_sample - 1

class MetaData:
    def __init__(self,parser, nsampes_per_data_type_dict):
        self.nsamples_dict = nsampes_per_data_type_dict
        self.parser = parser
