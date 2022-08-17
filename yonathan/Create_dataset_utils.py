import os
import numpy as np
import skimage.transform
from PIL import Image
import pickle
from Raw_data_loaders import *
from Create_dataset_classes import *
from Create_dataset_classes import ExampleClass



def Get_label_ordered(infos:list) -> np.array:
    """
    Args:
        infos: The information about all characters.

    Returns: list of all characters arranged in order.
    """
    
    row = list()
    rows = []  # A ll the rows.
    for k in range(len(infos)):  # Iterate for each character in the image.
        info = infos[k]
        char = info.label
        row.append(char)  # All the characters in the row.
        if info.edge_to_the_right:
            rows.append(row)
            row = list()
    label_ordered = np.array(rows)
    return label_ordered


def Get_label_existence(infos: list, nclasses: int) -> np.array:
    """
    Args:
        infos: The information about the sample.
        nclasses: The number of classes.

    Returns: The label_existence flag, telling for each entry whether the character is in the image.
    """
    label = np.zeros(nclasses)  # Initially all zeros.
    for info in infos:  # for each character in the image.
        label[info.label] = 1  # Set 1 in the label.
    label_existence = np.array(label)
    return label_existence



def store_sample_disk(sample:Sample, store_dir:str, folder_split:bool,folder_size:int):
    """
    Storing the sample on the disk.
    Args:
        sample: The sample we desire to save.
        store_dir: The directory we save in.
        folder_split: Whether we split the data into folders.
        folder_size: The folder size.

    """
    samples_dir =  store_dir
    i = sample.id
    if folder_split:
        # The inner path, based on the sample id.
        samples_dir = os.path.join(store_dir, '%d' % (i // folder_size))
        if not os.path.exists(samples_dir):
            os.makedirs(samples_dir, exist_ok=True)
    img = sample.image # Saving the image.
    img_fname = os.path.join(samples_dir, '%d_img.jpg' % i) # The image directory.
   # img = img.transpose((0, 1, 2))
    c = Image.fromarray(img.transpose(1,2,0)) # Convert to Image format.
    c.save(img_fname) # Saving the image.
    del sample.image, sample.infos
    # The labels directory.
    data_fname = os.path.join(samples_dir, '%d_raw.pkl' % i)
    # Dumping the labels in the pickle file.
    with open(data_fname, "wb") as new_data_file:
        pickle.dump(sample, new_data_file)
        

def AddCharacterToExistingImage(DataLoader:DataSet, image:np.array, char:CharInfo)->tuple:
    """
    Function Adding a character to a given image.
    Args:
        DataLoader: The raw data loader.
        image: The image we desire to add a character to.
        char: The character we desire to add.

    Returns: The new image and the info about the character.

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
    c = DataLoader.nchannels
    sz = scale * np.array(im.shape[1:])
    sz = sz.astype(np.int)
    h, w = sz
    sz = (c,*sz)
    im = skimage.transform.resize(im, sz, mode='constant') # Apply the transform.
    stx = char.location_x
    endx = stx + w
    sty = char.location_y
    endy = sty + h
    # this is a view into the image and when it changes the image also changes
    part = image[:,sty:endy, stx:endx] # The part of the image we plan to plant the character.
    mask = im.copy()
    mask[mask > 0] = 1
    mask[mask < 1] = 0
    rng = mask > 0
    # Change the part, yielding changing the original image.
    part[rng] = mask[rng] * im[rng] + (1 - mask[rng]) * part[rng]
    edge_to_the_right = char.edge_to_the_right
    info = CharInfo(label, label_id,  im, stx, endx,
                    sty, endy, edge_to_the_right)
    return image, info  # Return the image, the character info.

def Get_label_task(example:ExampleClass, infos:list,label_ordered:np.array,nclasses:int)->tuple:
    """
    Args:
        example: The sample example.
        infos: The information about all characters.
        label_ordered: The label_all.
        nclasses: The number of classes.

    Returns: The label_task, the flag.
    """
    query_part_id = example.query_part_id # The index we create about.
    info = infos[query_part_id] # The character information in the index.
    char = info.label # The label of the character.
    rows, obj_per_row = label_ordered.shape # The number of rows, the number of characters in evert row.
    adj_type = example.adj_type # The direction we query about.
    edge_class = nclasses # The label of the edge class is the number of characters.

    # Find the row and the index of the character.
    r, c = (label_ordered == char).nonzero()
    r = r[0]
    c = c[0]
    # find the adjacent char
    if adj_type == 0:
        # right
        # If in the border, the class is edge class.
        if c == (obj_per_row - 1):
            label_task = edge_class
        else:
            # Otherwise we return the character appearing in the right.
            label_task = label_ordered[r, c + 1]
    else:
        # left
        if c == 0:
            # If in the border, the class is edge class.
            label_task = edge_class
        else:
            label_task = label_ordered[r, c - 1] # Otherwise we return the character appearing in the left.
    flag = np.array([adj_type, char]) # The flag.
    return label_task,flag

def Get_valid_pairs_for_the_combinatorial_test(parser:argparse, nclasses:int,valid_classes:list,num_chars_to_sample)->tuple:
    """
    Exclude part of the training data. Validation set is from the train distribution. Test is only the excluded data (combinatorial generalization)
    How many strings (of nsample_chars) to exclude from training
    For 24 characters, 1 string excludes about 2.4% pairs of consecutive characters, 4 strings: 9.3%, 23 strings: 42%, 37: 52%
    For 6 characters, 1 string excludes about 0.6% pairs of consecutive characters, 77 strings: 48%, 120: 63%, 379: 90%
    these numbers were simply selected in order to achieve a certain percentage)
    Args:
        parser: The option parser.
        nclasses: The number of classes.
        valid_classes: The valid classes.
        num_chars_to_sample: Number of different sequences for the CG test.

    Returns: The valid pairs and the list of all chosen sequences.
    """
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
    for _ in range(ntest_strings):
        test_chars = np.random.choice(valid_classes, num_chars_to_sample, replace=False)
        print('test chars:', test_chars)
        test_chars_list.append(test_chars)
        test_chars = test_chars.reshape((-1,parser.nchars_per_row))
        for row in test_chars:
            for pi in range(parser.nchars_per_row- 1):
                cur_char = row[pi]
                adj_char = row[pi + 1]
                valid_pairs[cur_char, adj_char] = 0
                # now in each sample will be no pair which is pair in the sampled characters.
    # The number of available pairs.
    avail_ratio = valid_pairs.sum() / (len(valid_classes) * (len(valid_classes) - 1))
    exclude_percentage = (100 * (1 - avail_ratio))  # The percentage.
    print('Excluding %d strings, %f percentage of pairs' %
          (ntest_strings, exclude_percentage))
    # return the valid pairs, the test_characters.
    return valid_pairs, test_chars_list


def Get_sample_chars(prng:random, valid_pairs:np.array,is_test:bool,valid_classes:list,num_characters_per_sample:int, ntest_strings:int,test_chars_list:list)->list:
    """
    Returns a valid sample.
    If the sample is from the test dataset then the sample will be one of the test_chars_list.
    Otherwise it is build according to the CG rules.
    Args:
        prng: The random generator.
        valid_pairs: The valid pairs we can sample from.
        is_test: Whether the dataset is the test dataset.
        valid_classes: The valid classes we sample from.
        num_characters_per_sample: The number of characters we want to sample.
        ntest_strings: The number of strings in the CG test.
        test_chars_list: The list of all strings in the CG test.

    Returns: A sequence of characters.
    """
    
    sample_chars = []
    if not is_test:
        found = False
        while not found:
            # faster generation of an example than a random sample
            found_hide_many = False
            while not found_hide_many:
                sample_chars = []
                cur_char = prng.choice(valid_classes, 1)[0]
                sample_chars.append(cur_char)
                for _ in range(num_characters_per_sample - 1):
                    cur_char_adjs = valid_pairs[cur_char].nonzero()[0]
                    # create a permutation: choose each character at most once
                    cur_char_adjs = np.setdiff1d(cur_char_adjs, sample_chars)
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
                query_part_ids = prng.choice(
                    valid_queries, cur_ngenerate, replace=False)
        for query_part_id in query_part_ids:
            example = ExampleClass(sample_chars,query_part_id,adj_type,chars )
            examples.append(example)

# Only for emnist.


def Get_valid_classes(parser, nclasses):
    """
    :param parser: 
    :param nclasses: 
    :return: 
    """
    use_only_valid_classes = parser.use_only_valid_classes  # False #CHANGE IN EMNIST TO TRUE
    if use_only_valid_classes:
        # remove some characters which are very similar to other characters
        # ['O', 'F', 'q', 'L', 't', 'g', 'C', 'S', 'I', 'B', 'Y', 'n', 'b', 'X', 'r', 'H', 'P', 'G']
        invalid = [24, 15, 44, 21, 46, 41, 12, 28,
                   18, 11, 34, 43, 37, 33, 45, 17, 25, 16]
        # invalid=[i for i in range(47) if i not in [0,1,2,3,4,5,6,7,8,9] ]
        all_classes = np.arange(0, nclasses)
        valid_classes = np.setdiff1d(all_classes, invalid)
    else:
        valid_classes = np.arange(0, nclasses)
    return valid_classes


def Get_data_dir(parser, store_folder, language_list):
    """
    :param parser: 
    :param store_folder: 
    :param language_list: 
    :return: 
    """
    base_storage_dir = '%d_' % (parser.nchars_per_row *parser.num_rows_in_the_image)
    base_storage_dir += 'extended_' + str(language_list)
    store_dir = os.path.join(store_folder, 'new_samples')
    base_samples_dir = os.path.join(store_dir, base_storage_dir)
    if not os.path.exists(base_samples_dir):
        os.makedirs(base_samples_dir, exist_ok=True)
    storage_dir = base_samples_dir
    conf_data_fname = os.path.join(storage_dir, 'conf')
    return conf_data_fname, storage_dir
