import numpy as np
from torchvision import transforms
import argparse
import random
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from types import SimpleNamespace

class DsType:
    """
    Class holding all dataset types the code can handle.
    """
    def __init__(self,ds_name):
        assert ds_name in ['emnist','fashionmnist','omniglot','kmnist','SVHN']
        self.ds_name = ds_name

class Sample:
    #Class containing all information about the sample, including the image, the flags, the label task.
    def __init__(self, image, sample_id, label_existence, label_ordered, query_part_id, label_task, flag, is_train):
        """
        #Class containing all information about the sample, including the image, the flags, the label task.
        Args:
            infos: All information about all characters in the image.
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
            sample: The sample.
            query_part_id: The index we query about.
            adj_type: The direction we query about.
            chars: The list of all characters in the sample.
        """
        self.sampled_chars = sampled_chars
        self.query_part_id = query_part_id
        self.adj_type = adj_type
        self.chars = chars

class GetAugData:
    # Class returning for a given image size a data augmentation transform.
    def __init__(self,image_size:tuple):
        """
        Args:
            image_size: The image size.
        """
        # augmentation init
        self.aug_seed = 0
        color_add_range = int(0.2 * 255)
        rotate_deg = 2
        # translate by -20 to +20 percent (per axis))
        aug = iaa.Sequential( [iaa.Affine(  translate_percent={   "x": (-0.05, 0.05),  "y": (-0.1, 0.1) },  rotate=(-rotate_deg, rotate_deg), mode='edge', name='affine'), ], random_state=0)
        aug_nn_interpolation = aug.deepcopy()
        aff_aug = aug_nn_interpolation.find_augmenters_by_name('affine')[0]
        aff_aug.order = iap.Deterministic(0)
        self.aug_nn_interpolation = aug_nn_interpolation
        # this will be used to add random color to the segmented avatar but not the segmented background.
        aug.append(iaa.Add(  (-color_add_range,color_add_range)))  # only for the image not the segmentation
        self.aug = aug
        self.image_size = image_size

class DataAugmentClass:
    """
    class performing the augmentation.
    """
    def __init__(self, image:np.array, label_existence:np.array, aug_data:transforms,augment:bool):
        """
        Args:
            image: The image we want to augment.
            label_existence: The label existence.
            aug_data: The data augmentation transform.
        """
        self.images = image[np.newaxis]
        self.labels = label_existence[np.newaxis]
        self.batch_range = range(1)
        self.aug_data = aug_data
        self.augment = augment

    def get_batch_base(self):
        batch_range = self.batch_range
        batch_images = self.images[batch_range]

        aug_data = self.aug_data
        augment_type = 'aug_package'
        if self.augment:
            aug_seed = aug_data.aug_seed
            if augment_type == 'aug_package':
                aug = aug_data.aug
                aug.seed_(aug_seed)
                batch_images = aug.augment_images(batch_images)
                aug_data.aug_seed += 1
        return batch_images[0]

class CharacterTransforms:
    def __init__(self,parser:argparse, prng:random,label_ids:list,samplei:int,sample_chars:int ):
        """
        Args:
            parser: The options parser.
            prng: The random generator.
            label_ids: The label ids list.
            samplei: The sample id.
            sample_chars: The sampled characters list
        """
        minscale = 1.0
        maxscale = 1.5
        minshift = 2.0
        maxshift = .2 * parser.letter_size
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
    def __init__(self,parser, nsamples_train, nsamples_test, nsamples_val, valid_classes):
        self.nsamples_train = nsamples_train
        self.nsamples_test = nsamples_test
        self.nsamples_val = nsamples_val
        self.parser = parser
        self.valid_classes = valid_classes