import numpy as np
from torchvision import transforms
import argparse
import random
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from types import SimpleNamespace
import albumentations as A

# import cv2
# cv2_BORDER_CONSTANT = cv2.BORDER_CONSTANT
cv2_BORDER_CONSTANT = 0


class CharInfo:
    def __init__(self,label:int, label_idx:int, im:np.array, stx:int, endx:int, sty:int, endy:int, edge_to_the_right:bool):
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
    #Class containing all information about the sample, including the image, the flags, the label task.
    def __init__(self, infos, image, label_existence, label_ordered,sample_id, query_part_id,flag, label_task, keypoints):
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
        self.keypoints = keypoints

class ExampleClass:
    def __init__(self,sample:Sample,query_part_id:int,adj_type:int,chars:list):
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
        aug = iaa.Sequential( [iaa.Affine(  translate_percent={   "x": (-0.1, 0.1),  "y": (-0.05, 0.05) },  rotate=(-rotate_deg, rotate_deg), mode='edge', name='affine'), ], random_state=0)
        aug_nn_interpolation = aug.deepcopy()
        aff_aug = aug_nn_interpolation.find_augmenters_by_name('affine')[0]
        aff_aug.order = iap.Deterministic(0)
        self.aug_nn_interpolation = aug_nn_interpolation
        # this will be used to add random color to the segmented avatar but not the segmented background.
        aug.append(iaa.Add(  (-color_add_range,color_add_range)))  # only for the image not the segmentation
        self.aug = aug
        self.image_size = image_size


def augment_albumentations(img, aug_data, keypoints=None):
    # seed = aug_data.seed
    rotate_deg = aug_data.rotate_deg
  #  color_add_range = aug_data.color_add_range
    # xtrans  = aug_data.xtrans
    # ytrans  = aug_data.ytrans
    xtrans = 0.05
    ytrans = 0.05
    # A.Rotate(border_mode=cv2_BORDER_CONSTANT,value=(255,255,255),p=1),
    ssr = A.ShiftScaleRotate(shift_limit=xtrans, scale_limit=0, rotate_limit=rotate_deg, interpolation=1,
                              value=(0, 0, 0), mask_value=(0, 0, 0),
                             shift_limit_x=None, shift_limit_y=None, always_apply=False, p=1)
    rbc = A.RandomBrightnessContrast(p=1)
    if keypoints is not None:
       # keypoints = [keypoints]
        keypoint_params = A.KeypointParams(format='xy', remove_invisible=False, angle_in_degrees=True)
    else:
        keypoint_params = None

    transform = A.Compose([ssr, rbc], keypoint_params=keypoint_params)

    transformed = transform(image=img, keypoints=keypoints)
    transformed_image = transformed["image"]
    if keypoints != None:
     transformed_keypoints = transformed["keypoints"]
    else:
     transformed_keypoints = None
    return transformed_image, transformed_keypoints


def get_aug_data(IMAGE_SIZE):
    aug_data = SimpleNamespace()
    aug_data.color_add_range = int(0.2 * 255)
    aug_data.rotate_deg = 10
    aug_data.xtrans = 0.1 * IMAGE_SIZE[1]
    aug_data.ytrans = 0.05 * IMAGE_SIZE[0]
    aug_data.image_size = IMAGE_SIZE
    return aug_data

class DataAugmentClass:
    """
    class performing the augmentation.
    """
    def __init__(self, image:np.array, label_existence:np.array, aug_data:transforms):
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

    def get_batch_base(self):
        batch_data = get_batch_base(self)
        image = batch_data.images[0]
        return image

def get_batch_base(aug_data_struct:DataAugmentClass)->SimpleNamespace:
    """
    Args:
        aug_data_struct: The augmentation data struct.

    Returns: Struct containing images, labels.

    """
    batch_range = aug_data_struct.batch_range
    batch_images =aug_data_struct.images[batch_range]
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
   # def __init__(self, prng: random, letter_size: int, label_ids: list, samplei: int, sample_chars: int, total_rows:,   obj_per_row, IMAGE_SIZE, sample_nchars):
    def __init__(self,parser:argparse, prng:random,label_ids:list,samplei:int,sample_chars:int ):
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
        self.letter_size = parser.letter_size
        new_size = int(self.scale * parser.letter_size)
        shift = prng.rand(2) * (maxshift - minshift) + minshift
        y, x = shift.astype(np.int)
        origr, origc = np.unravel_index(samplei, [parser.num_rows_in_image, parser.nchars_per_row])
        c = origc + 1  # start from column 1 instead of 0
        r = origr
        _ , imageh , imagew = parser.image_size
        stx = c * parser.letter_size + x
        stx = max(0, stx)
        stx = min(stx, imagew - new_size)
        sty = r * parser.letter_size + y
        sty = max(0, sty)
        sty = min(sty, imageh - new_size)
        self.location_x = stx
        self.location_y = sty
        midx = int(np.rint(self.location_x + (self.scale * self.letter_size)/2))
        midy = int(np.rint(self.location_y + (self.scale * self.letter_size)/2))
        self.middle_point = (midx,midy)
       # self.middle_point = int(self.middle_point)
        self.edge_to_the_right = origc == parser.nchars_per_row - 1 or samplei == parser.sample_nchars - 1

    def update(self):
        return None

class Meta_data:
    def __init__(self, nsamples_train, nsamples_test, nsamples_val, nclasses_per_language, letter_size, image_size, num_rows_in_image, obj_per_row):
        self.nsamples_train = nsamples_train
        self.nsamples_test = nsamples_test
        self.nsamples_val = nsamples_val
        self.nclasses_per_language = nclasses_per_language
        self.letter_size = letter_size
        self.image_size = image_size
        self.num_rows_in_image = num_rows_in_image
        self.obj_per_row = obj_per_row
        self.nchars_per_image = num_rows_in_image * obj_per_row