import torch.utils.data as data
from torchvision import transforms
import os
import torch
from skimage import io
import numpy as np
import skimage.transform
from skimage import color
from types import SimpleNamespace
from PIL import Image
import pickle
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import imgaug as ia
import argparse
import random

def folder_size(path: str)->int:
 """
 :param path: path to the raw data.
 :returns The size of a given folder.
 """
 size = 0
 for _ in os.scandir(path):
  size+=1
 return size

def create_dict(path: str)->dict:
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

def Get_label_ordered(infos:list)->np.array:
    """
    :param infos: The information about the sample.
    :return: Returns the label_all flag.
    """
    row = list()
    rows = [] #A ll the rows.
    for k in range(len(infos)): # Iterate for each character in the image.
        info = infos[k]
        char = info.label
        row.append(char) # All the characters in the row.
        if info.edge_to_the_right:
            rows.append(row)
            row = list()
    label_ordered = np.array(rows)
    return label_ordered


def Get_label_existence(infos:list, nclasses:int)->np.array:
    """
    :param infos: The information about the sample.
    :param nclasses: The number of classes.
    :Returns the label_existence flag, telling for each entry whether the character is in the image.
    """
    label = np.zeros(nclasses) # Initially all zeros.
    for info in infos: # for each character in the image.
        label[info.label] = 1 # Set 1 in the label.
    label_existence = np.array(label)
    return label_existence

class CharInfo():
    """
    ImageInfo class.
    """
    def __init__(self,label:int, label_idx:int, im:np.array, stx:int, endx:int, sty:int, endy:int, edge_to_the_right:bool)->None:
        """
        :param label: The character label.
        :param label_idx: The character label index telling its index in the number of characters with the same label.
        :param im: The charcter image.
        :param mask: # Not clear-TODO -CHECK IT.
        :param stx: The beginning place in the x-axis.
        :param endx: The end place in the x-axis.
        :param sty: The beginning place in the y-axis.
        :param endy: The end place in the y-axis.
        :param edge_to_the_right: boolean flag telling whether the character is near the border.
        """
        self.label = label
        self.label_idx = label_idx
        self.im = im
        self.stx = stx
        self.endx = endx
        self.sty = sty
        self.endy = endy
        self.edge_to_the_right = edge_to_the_right
        
class example_class():
    def __init__(self,sample,query_part_id,adj_type,chars):
        self.sample = sample
        self.query_part_id = query_part_id
        self.adj_type = adj_type
        self.chars = chars
        
class Sample:
    #Class containing all information about the sample, including the image, the flags, the label task.
    def __init__(self, infos, image, sample_id, label_existence, label_ordered, query_part_id, label_task, flag, is_train):
        """
        :param infos: All information about all characters in the image.
        :param image: The image.
        :param sample_id: # The id.
        :param label_existence: The label existence.
        :param label_ordered: The label ordered, containing all the characters arranged by the order.
        :param query_part_id: The index we query about.
        :param label_task: The label task.
        :param flag: The flag.
        :param is_train: Whether the sample is part of the training set.
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

class get_aug_data():
    # Class returning for a given image size a data augmentation transform.
    def __init__(self,IMAGE_SIZE:tuple)->None:
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
        # this will be used to add random color to the segmented avatar but not the segmented background
        aug.append(iaa.Add(  (-color_add_range,color_add_range)))  # only for the image not the segmentation
        self.aug = aug
        self.image_size = IMAGE_SIZE

def store_sample_disk(sample, store_dir, folder_split,folder_size):
    """
     Storing the samples on the disk.
    :param sample: The sample we desire to save.
    :param store_dir: The directory we save in.
    :param folder_split: Whether we split the data into folders.
    :param folder_size: The folder size.
    """
    samples_dir =  store_dir
    i = sample.id
    if folder_split:
        samples_dir = os.path.join(store_dir, '%d' % (i // folder_size)) # The inner path, based on the sample id.
        if not os.path.exists(samples_dir):
            os.makedirs(samples_dir, exist_ok=True)
    img = sample.image # Saving the image.
    img_fname = os.path.join(samples_dir, '%d_img.jpg' % i) # The image directory.
    c = Image.fromarray(img) # Convert to Image format.
    c.save(img_fname) # Saving the image.
    del sample.image, sample.infos
    data_fname = os.path.join(samples_dir, '%d_raw.pkl' % i) # The labels directory.
    with open(data_fname, "wb") as new_data_file: # Dumping the labels in the pickle file.
        pickle.dump(sample, new_data_file)

class DataAugmentClass:
    """
    class performing the augmentation.
    """
    def __init__(self, image, label_existence, aug_data):
        """
        :param image: The image we want to augment.
        :param label_existence: The label existence.
        :param aug_data:
        """
        self.images = image[np.newaxis]
        self.labels = label_existence[np.newaxis]
        self.batch_range = range(1)
        self.aug_data = aug_data

    def get_batch_base(self):
        batch_data = get_batch_base(self)
        image = batch_data.images[0]
        return image

def get_batch_base(Aug_data_struct):
    """
    :param Aug_data_struct:
    :return:
    """
    batch_range = Aug_data_struct.batch_range
    batch_images = Aug_data_struct.images[batch_range]
    batch_labels = Aug_data_struct.labels[batch_range]

    aug_data = Aug_data_struct.aug_data
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
 #TODO CHECK FOR MULTIPLE LNAGUGES
 
class OmniglotDataLoader(data.Dataset):
    """
    Return data loader for omniglot.
    """
    def __init__(self, languages:list, Raw_data_source='/home/sverkip/data/omniglot/data/omniglot_all_languages'):
        """
        :param languages: The languages we desire to load.
        :param data_path:
        """
        images = [] # The images list.
        labels = [] # The labels list.
        sum, num_charcters = 0.0,0
        transform = transforms.Compose( [transforms.ToTensor(), transforms.functional.invert, transforms.Resize([28, 28])]) # Transforming to tensor + resizing.
        Data_source = '/home/sverkip/data/omniglot/data/omniglot_all_languages'
        dictionary = create_dict(Raw_data_source) # Getting the dictionary.
        language_list = os.listdir(Raw_data_source) # The languages list.
        for language_idx in languages: # Iterating over all the list of languages.
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
                num_charcters = num_charcters +1
            sum = sum + dictionary[language_idx]
        images = torch.stack(images, dim=0).squeeze()
        labels = torch.stack(labels, dim=0)
        self.images = images
        self.labels = labels
        self.nclasses = num_charcters
        self.num_examples_per_character = len(labels)//num_charcters

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        image = image
        return image, label

  # Function Adding a character to a given image.


def AddCharacterToExistingImage(OmniglotDataLoader:OmniglotDataLoader, image:np.array, char:CharInfo, CHAR_SIZE:list, num_examples_per_character:int)->None:
    """
    :param OmniglotDataLoader: The Omniglot data-loader.
    :param image: The image we desire to add a character to.
    :param char: The information about the character we desire to add.
    :param PERSON_SIZE: The character size.
    :param num_examples_per_character: Number of characters with the same label.
    :return: The image after the new character was added and the info about the character.
    """

    label= char.label # The label of the character.
    label_id = char.label_id # The label -d.
    img_id = num_examples_per_character * label + label_id  # The index in the data-loader.
    im, _ = OmniglotDataLoader[img_id] # The character image.
    scale = char.scale
    sz = scale * np.array([CHAR_SIZE, CHAR_SIZE])
    sz = sz.astype(np.int)
    h, w = sz
    im = skimage.transform.resize(im, sz, mode='constant') # Apply the transform.
    stx = char.location_x
    endx = stx + w
    sty = char.location_y
    endy = sty + h
    # this is a view into the image and when it changes the image also changes
    part = image[sty:endy, stx:endx] # The part of the image we plan to plant the character.
    mask = im.copy()
    mask[mask > 0] = 1
    mask[mask < 1] = 0
    rng = mask > 0
    part[rng] = mask[rng] * im[rng] + (1 - mask[rng]) * part[rng] # Change the part, yielding changing the original image.
    edge_to_the_right = char.edge_to_the_right
    info = CharInfo(label, label_id,  im, stx, endx, sty, endy, edge_to_the_right)
    return image, info # Return the image, the character info.


def Get_label_task(example:example_class, infos:list,label_ordered:np.array,nclasses:int)->tuple:
    """
    :param example: The example.
    :param infos: The information about all characters.
    :param label_ordered: The label of all characters arranged.
    :param nclasses: The number of classes in the data-set.
    :return: The label task,The flag for the TD network, the index we query about.
    """
    query_part_id = example.query_part_id # The index we create about.
    info = infos[query_part_id] # The character information in the index.
    char = info.label # The label of the character.
    rows, obj_per_row = label_ordered.shape # The number of rows, the number of characters in evert row.
    adj_type = example.adj_type # The direction we query about.
    edge_class = nclasses # The label of the edge class is the number of characters.

    r, c = (label_ordered == char).nonzero() # Find the row and the index of the character.
    r = r[0]
    c = c[0]
    # find the adjacent char
    if adj_type == 0:
        # right
        if c == (obj_per_row - 1): # If in the border, the class is edge class.
            label_task = edge_class
        else:
            label_task = label_ordered[r, c + 1] # Otherwise we return the character appearing in the right.
    else:
        # left
        if c == 0:
            label_task = edge_class # If in the border, the class is edge class.
        else:
            label_task = label_ordered[r, c - 1] # Otherwise we return the character appearing in the left.
    flag = np.array([adj_type, char]) # The flag.
    return label_task,flag, query_part_id # return.

def Get_valid_pairs_for_the_combenatorial_test(parser:argparse, nclasses:int,valid_classes:list,num_chars_to_sample,num_charcters_per_row):
    """
    :param nclasses: The number of classes.
    :param valid_classes: The valid classes we can choose from.
    :param sample_nchars: The number of characters to sample.
    :param num_characters_per_row: The number of characters in each row.
    :return: valid_pairs the valid pairs, test_chars_list- the characters for the test dataset.
    """
    # Exclude part of the training data. Validation set is from the train distribution. Test is only the excluded data (combinatorial generalization)

    # How many strings (of nsample_chars) to exclude from training
    # For 24 characters, 1 string excludes about 2.4% pairs of consecutive characters, 4 strings: 9.3%, 23 strings: 42%, 37: 52%
    # For 6 characters, 1 string excludes about 0.6% pairs of consecutive characters, 77 strings: 48%, 120: 63%, 379: 90%
    # (these numbers were simply selected in order to achieve a certain percentage)
   # ntest_strings = 1
    ntest_strings = parser.ntest_strings
    # generate once the same test strings
    np.random.seed(777)
    valid_pairs = np.zeros((nclasses, nclasses))
    for i in valid_classes:
        for j in valid_classes:
            if j != i:
                valid_pairs[i, j] = 1

    test_chars_list = []
    # it is enough to validate only right-of pairs even when we query for left of as it is symmetric
    for i in range(ntest_strings):
        test_chars = np.random.choice(valid_classes, num_chars_to_sample, replace=False)
        print('test chars:', test_chars)
        test_chars_list.append(test_chars)
        test_chars = test_chars.reshape((-1,num_charcters_per_row))
        for row in test_chars:
            for pi in range(num_charcters_per_row - 1):
                cur_char = row[pi]
                adj_char = row[pi + 1]
                valid_pairs[cur_char, adj_char] = 0
                # now in each sample will be no pair which is pair in the sampled characters.
    avail_ratio = valid_pairs.sum() / (len(valid_classes) *  (len(valid_classes) - 1)) # The number of available pairs.
    exclude_percentage = (100 * (1 - avail_ratio)) # The percentage.
    print('Excluding %d strings, %f percentage of pairs' % (ntest_strings, exclude_percentage))
    return valid_pairs,test_chars_list # return the valid pairs, the test_characters.

def Get_sample_chars(prng:random, valid_pairs:np.array,is_test:bool,valid_classes,num_characters_per_sample, ntest_strings,test_chars_list):
    # Returns a sample if the CG is turned on.
    """
    :param prng: The prng.
    :param valid_pairs: The valid pairs we can sample from.
    :param is_test: Whether the dataset is the test dataset.
    :param valid_classes: The valid classes we sample from.
    :param num_characters_per_sample: The number of characters we want to sample.
    :param ntest_strings: The number of strings in the CG test.
    :param test_chars_list: The list of all strings in the CG test.
    :return: The sampled chars.
    """
    if is_test:
        found = False
        while not found:
            # faster generation of an example than a random sample
            found_hide_many = False
            while not found_hide_many:
                sample_chars = []
                cur_char = prng.choice(valid_classes, 1)[0]
                sample_chars.append(cur_char)
                for pi in range(num_characters_per_sample - 1):
                    cur_char_adjs = valid_pairs[cur_char].nonzero()[0]
                    cur_char_adjs = np.setdiff1d(cur_char_adjs, sample_chars)  # create a permutation: choose each character at most once
                    if len(cur_char_adjs) > 0:
                        cur_adj = prng.choice(cur_char_adjs, 1)[0]
                        cur_char = cur_adj
                        sample_chars.append(cur_char)
                        if len(sample_chars) == num_characters_per_sample:
                            found_hide_many = True
                    else:
                        break

            found = True
    else:
        test_chars_idx = prng.randint(ntest_strings)
        sample_chars = test_chars_list[test_chars_idx]
    return sample_chars

class CharcterTransforms():
    def __init__(self,prng:random,LETTER_SIZE,label_ids,samplei,sample_chars,total_rows,obj_per_row,IMAGE_SIZE,sample_nchars ):
    
        """
        :param prng:
        :param LETTER_SIZE:
        :param label_ids:
        :param samplei:
        :param sample_chars:
        :param total_rows:
        :param obj_per_row:
        :param IMAGE_SIZE:
        :param sample_nchars:
        """
        minscale = 1
        maxscale = 1.5
        minshift = 0
        maxshift = .2 * LETTER_SIZE
        # place the chars on the image
        self.label_id = label_ids[samplei]
        self.label = sample_chars[samplei]
        self.scale = prng.rand() * (maxscale - minscale) + minscale
        new_size = int(self.scale * LETTER_SIZE)

        shift = prng.rand(2) * (maxshift - minshift) + minshift
        y, x = shift.astype(np.int)
        origr, origc = np.unravel_index(samplei, [total_rows, obj_per_row])
        c = origc + 1  # start from column 1 instead of 0
        r = origr
        imageh, imagew = IMAGE_SIZE
        stx = c * LETTER_SIZE + x
        stx = max(0, stx)
        stx = min(stx, imagew - new_size)
        sty = r * LETTER_SIZE + y
        sty = max(0, sty)
        sty = min(sty, imageh - new_size)
        self.location_x = stx
        self.location_y = sty
        self.edge_to_the_right = origc == obj_per_row - 1 or samplei == sample_nchars - 1



def create_examples_per_sample(examples,sample_chars,chars, prng, adj_types,ncharacters_per_image,single_feat_to_generate,is_test,is_val,ngenerate):
    """
    :param examples: The sample examples list.
    :param sample_chars: The sampled characters list.
    :param chars: The list of all information about all h
    :param prng: 
    :param adj_types: 
    :param ncharacters_per_image: 
    :param single_feat_to_generate: 
    :param is_test: 
    :param is_val: 
    :param ngenerate: s
    :return: 
    """
    for adj_type in adj_types:
        valid_queries = range(ncharacters_per_image)
        if single_feat_to_generate or is_test or is_val:
            query_part_ids = [prng.choice(valid_queries)]
        else:
            cur_ngenerate = ngenerate

            if cur_ngenerate == -1:
                query_part_ids = valid_queries
            else:
                query_part_ids = prng.choice(valid_queries, cur_ngenerate, replace=False)
        for query_part_id in query_part_ids:
            example = example_class(sample_chars,query_part_id,adj_type,chars )
            examples.append(example)
            
#Only for emnist.

def Get_valid_classes(parser,nclasses):
    """
    :param parser: 
    :param nclasses: 
    :return: 
    """
    use_only_valid_classes = parser.use_only_valid_classes  # False #CHANGE IN EMNIST TO TRUE
    if use_only_valid_classes:
        # remove some characters which are very similar to other characters
        # ['O', 'F', 'q', 'L', 't', 'g', 'C', 'S', 'I', 'B', 'Y', 'n', 'b', 'X', 'r', 'H', 'P', 'G']
        invalid = [24, 15, 44, 21, 46, 41, 12, 28, 18, 11, 34, 43, 37, 33, 45, 17, 25, 16]
        # invalid=[i for i in range(47) if i not in [0,1,2,3,4,5,6,7,8,9] ]
        all_classes = np.arange(0, nclasses)
        valid_classes = np.setdiff1d(all_classes, invalid)
    else:
        valid_classes = np.arange(0, nclasses)
    return valid_classes

def Get_data_dir(parser, store_folder,language_list):
    """
    :param parser: 
    :param store_folder: 
    :param language_list: 
    :return: 
    """
   # store_folder = '/home/sverkip/data/omniglot/data'
    base_storage_dir = '%d_' % (parser.nchars_per_row *parser.num_rows_in_the_image)
    base_storage_dir += 'extended_test' + str(language_list)
    store_dir = os.path.join(store_folder, 'new_samples')
    base_samples_dir = os.path.join(store_dir, base_storage_dir)
    if not os.path.exists(base_samples_dir):
        os.makedirs(base_samples_dir, exist_ok=True)
    storage_dir = base_samples_dir
    conf_data_fname = os.path.join(storage_dir, 'conf')
    return conf_data_fname, storage_dir