import numpy as np
from torchvision import transforms
import argparse
import random
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from types import SimpleNamespace


class CharInfo:
    """
     _summary_
    """    
    def __init__(self, label: int, label_idx: int, im: np.array, stx: int, endx: int, sty: int, endy: int, edge_to_the_right: bool):
        """
        ImageInfo class.
        Args:
            label: The character label.
            label_idx: he character label index telling its index in the number of characters with the same label.
            im: The character image.
            stx: The beginning place in the x-axis.
            endx: The end place in the x-axis.
            sty: The beginning place in the y-axis.
            endy: The end place in the y-axis.
            edge_to_the_right: boolean flag telling whether the character is near the border.
        """
        self.label = label
        self.label_idx = label_idx
        self.im = im
        self.stx = stx
        self.endx = endx
        self.sty = sty
        self.endy = endy
        self.edge_to_the_right = edge_to_the_right


class Sample:
    """
     _summary_
        This class represent a single sample in the dataset
        
        TODO: add also method to this class [create_example?]
        TODO: when creating the inputs, we still call them as a unstructured and use the inputs_to_struct method
        
        Attributes:
            infos: All information about all characters in the image.
            image: The image.
            id: The id.
            label_existence: The label existence.
            label_ordered: The label ordered, containing all the characters arranged.
            query_part_id: The index we query about.
            label_task: The label task.
            flag: The flag.
            is_train: Whether the sample is part of the training set.


    """
    # Class containing all information about the sample, including the image, the flags, the label task.

    def __init__(self, infos, image, sample_id, label_existence, label_ordered, query_part_id, label_task, flag, is_train):
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
        self.infos = infos
        self.image = image
        self.id = sample_id
        self.label_existence = label_existence.astype(np.int)
        self.label_ordered = label_ordered
        self.query_part_id = query_part_id
        self.label_task = label_task
        self.flag = flag
        self.is_train = is_train


class ExampleClass:
    """
     _summary_


     #TODO what is the difference between this and the above class?
    """

    def __init__(self, sample: Sample, query_part_id: int, adj_type: int, chars: list):
        """
        Args:
            sample: The sample.
            query_part_id: The index we query about.
            adj_type: The direction we query about.
            chars: The list of all characters in the sample.
        """
        self.sample = sample
        self.query_part_id = query_part_id
        self.adj_type = adj_type
        self.chars = chars


class GetAugData:
    """
     _summary_
    """
    # Class returning for a given image size a data augmentation transform.

    def __init__(self, image_size: tuple):
        """
        Args:
            image_size: The image size.
        """
        # augmentation init
        self.aug_seed = 0
        color_add_range = int(0.2 * 255)
        rotate_deg = 2
        # translate by -20 to +20 percent (per axis))
        aug = iaa.Sequential([iaa.Affine(translate_percent={
                             "x": (-0.1, 0.1),  "y": (-0.05, 0.05)},  rotate=(-rotate_deg, rotate_deg), mode='edge', name='affine'), ], random_state=0)
        aug_nn_interpolation = aug.deepcopy()
        aff_aug = aug_nn_interpolation.find_augmenters_by_name('affine')[0]
        aff_aug.order = iap.Deterministic(0)
        self.aug_nn_interpolation = aug_nn_interpolation
        # this will be used to add random color to the segmented avatar but not the segmented background.
        # only for the image not the segmentation
        aug.append(iaa.Add((-color_add_range, color_add_range)))
        self.aug = aug
        self.image_size = image_size


class DataAugmentClass:
    """
    class performing the augmentation.
    #TODO thic class only holds the information about the augmentation
    #  Is it relevant? the aug_data object is rich enough for that...
    # maybe expand the existing class
    """

    def __init__(self, image: np.array, label_existence: np.array, aug_data: transforms):
        """
        Args:
            image: The image we want to augment.
            label_existence: The label existence.
            aug_data: The data augmentation transform.
        """
        self.images = image[np.newaxis]
        self.labels = label_existence[np.newaxis]
        self.batch_range = range(1) #what?
        self.aug_data = aug_data

    def get_batch_base(self):
        batch_data = get_batch_base(self)
        image = batch_data.images[0]
        return image


def get_batch_base(aug_data_struct: DataAugmentClass) -> SimpleNamespace:
    """
    Args:
        aug_data_struct: The augmentation data struct.

    Returns: Struct containing images, labels.

    """
    batch_range = aug_data_struct.batch_range
    batch_images = aug_data_struct.images[batch_range]
    batch_labels = aug_data_struct.labels[batch_range]

    aug_data = aug_data_struct.aug_data
    augment_type = 'aug_package'
    if aug_data.augment: 
        aug_seed = aug_data.aug_seed
        if augment_type == 'aug_package':
            aug = aug_data.aug
            aug.seed_(aug_seed)
            batch_images = aug.augment_images(batch_images)
            aug_data.aug_seed += 1

    result = SimpleNamespace()
    result.images = batch_images
    result.labels = batch_labels
    result.size = len(batch_images)
    return result


class CharacterTransforms:
    """
     _summary_
    """    
   # def __init__(self, prng: random, letter_size: int, label_ids: list, samplei: int, sample_chars: int, total_rows:,   obj_per_row, IMAGE_SIZE, sample_nchars):
    def __init__(self, parser: argparse, prng: random, label_ids: list, samplei: int, sample_chars: int):
        """
        Args:
            parser: The options parser.
            prng: The random generator.
            label_ids: The label ids list.
            samplei: The sample id.
            sample_chars: The sampled characters list
        """
        minscale = 1
        maxscale = 1.5
        minshift = 0
        maxshift = .2 * parser.letter_size
        # place the chars on the image
        self.label_id = label_ids[samplei]
        self.label = sample_chars[samplei]
        self.scale = prng.rand() * (maxscale - minscale) + minscale
        new_size = int(self.scale * parser.letter_size)

        shift = prng.rand(2) * (maxshift - minshift) + minshift
        y, x = shift.astype(np.int)
        origr, origc = np.unravel_index(
            samplei, [parser.num_rows_in_the_image, parser.nchars_per_row])
        c = origc + 1  # start from column 1 instead of 0
        r = origr
        _, imageh, imagew = parser.image_size
        stx = c * parser.letter_size + x
        stx = max(0, stx)
        stx = min(stx, imagew - new_size)
        sty = r * parser.letter_size + y
        sty = max(0, sty)
        sty = min(sty, imageh - new_size)
        self.location_x = stx
        self.location_y = sty
        self.edge_to_the_right = origc == parser.nchars_per_row - \
            1 or samplei == parser.sample_nchars - 1
