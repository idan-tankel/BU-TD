import torch.utils.data as data
import os
import torch
from skimage import io
import numpy as np
import skimage.transform
from skimage import color
from PIL import Image
import pickle
import imgaug as ia
from Raw_data import *
from Create_dataset_classes import *
import matplotlib.pyplot as plt
import datetime
import sys
from multiprocessing import Pool
import shutil



def store_sample_disk(parser,sample:Sample, store_dir:str):
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
    if parser.folder_split:
        samples_dir = os.path.join(store_dir, '%d' % (i // parser.folder_size)) # The inner path, based on the sample id.
        if not os.path.exists(samples_dir):
            os.makedirs(samples_dir, exist_ok=True)
    img = sample.image # Saving the image.
    img_fname = os.path.join(samples_dir, '%d_img.jpg' % i) # The image directory.
   # img = img.transpose((0, 1, 2))
    c = Image.fromarray(img.transpose(1,2,0)) # Convert to Image format.
    c.save(img_fname) # Saving the image.
    data_fname = os.path.join(samples_dir, '%d_raw.pkl' % i) # The labels directory.
    with open(data_fname, "wb") as new_data_file: # Dumping the labels in the pickle file.
        pickle.dump(sample, new_data_file)

def Create_raw_examples(parser,image_ids,k,ds_type,cur_nexamples,valid_pairs, valid_classes, test_chars_list, num_examples_per_character ):
    prng = np.random.RandomState(k)
    num_chars_per_image = parser.nchars_per_row * parser.num_rows_in_the_image
    examples = []
    for i in range(cur_nexamples):  # Iterating over the number of samples.
        if parser.generalize:  # If generalize we get the possible pairs we can choose from.
            sample_chars = Get_sample_chars(parser,prng, valid_pairs, ds_type, valid_classes, num_chars_per_image,  ntest_strings, test_chars_list)
        else:
            # Otherwise we choose from all the valid classes, without replacement the desired number of characters.
            sample_chars = prng.choice(valid_classes, num_chars_per_image, replace=False)
        image_id = []
        label_ids = []
        # For each character, sample an id in the number of characters with the same label.
        for _ in sample_chars:
            label_id = prng.randint(0, num_examples_per_character)  # Choose a possible label_id.
            image_id.append(label_id)
            label_ids.append(label_id)
        image_id_hash = str(image_id)
        if image_id_hash in image_ids:
            continue
        image_ids.add(image_id_hash)
        # place the chars on the image
        chars = []  # The augmented characters.
        for samplei in range(num_chars_per_image):  # For each chosen character, we augment it and transform it.
            char = CharacterTransforms(parser, prng, label_ids, samplei, sample_chars)
            chars.append(char)
        print(i)
        if parser.create_all_directions:  # Creating the possible tasks
            avail_adj_types = range(ndirections)
        else:
            avail_adj_types = [0]
        adj_types = [prng.choice(avail_adj_types)]
        create_examples_per_sample(parser,prng, ds_type, examples, sample_chars, chars, adj_types)
    return examples

# TODO-assert nclasses is the correct one.
def gen_sample(parser: argparse, sample_id: int, ds_type: str, aug_data: transforms, dataloader: DataSet, example: ExampleClass) -> Sample:
    """
    Creates a single sample including image, label_task, label_all, label_existence, query_index
    Args:
        parser: The option parser.
        sample_id: The sample id in all samples.
        is_train: If the sample is for training.
        aug_data: The data augmentation transform.
        dataloader: The raw data loader.
        example: The example containing the selected characters.
        augment_sample: Whether to augment the sample, true for train otherwise false.

    Returns: A sample.
    """
    # start by creating the image background(all black)
    augment_sample = parser.augment_sample
    is_train = ds_type == 'train'
    image = 0 * np.ones(parser.image_size, dtype=np.float32)
    infos = []  # Stores all the information about the characters.
    keypoints = []
    for char in example.chars:  # Iterate over each chosen character.
        image, info = AddCharacterToExistingImage(dataloader, image, char)  # Adding the character to the image.
        infos.append(info)  # Adding to the info about the characters.
        keypoints.append(char.middle_point)
    # Making label_existence flag.
    label_existence = Get_label_existence(infos, dataloader.nclasses)
    # the characters in order as seen in the image
    label_ordered = Get_label_ordered(infos)
    # instruction and task label
    # even for grayscale images, store them as 3 channels RGB like
    if image.shape[0] == 1:
        image = np.concatenate((image, image, image), axis=0)
    # Making RGB.
    image = image * 255
    image = image.astype(np.uint8)
    # Doing data augmentation
    label_task, flag, keypoint = Get_label_task(example, infos, label_ordered, dataloader.nclasses, keypoints)

    if is_train and augment_sample:
        # augment
        data_augment = DataAugmentClass(image, label_existence, aug_data, augment_sample)
        image = data_augment.get_batch_base()
    # Storing the needed information about the sample.
    sample = Sample(infos, image, sample_id, label_existence, label_ordered, example.query_part_id, label_task, flag, is_train, keypoint)
    return sample  # Returning the sample we are going to store.

def gen_samples(parser: argparse, dataloader: DataSet, job_id: int, range_start: int, range_stop: int, examples: list, storage_dir: str, ds_type: str) -> None:
    """
    Generates and stored samples, by calling to create_sample and store_sample_disk_pytorch.
    Args:
        parser: The option parser.
        dataloader: The raw data loader.
        job_id: The job id.
        range_start: The range start in the job.
        range_stop: The range stop of the job.
        examples: The chosen examples.
        storage_dir: The storage directory
        ds_type: The data-set type.
        augment_sample: Whether to augment the sample.

    """
    image_size = parser.image_size  # The image size.
    aug_data = None  # The augmentation transform.
    is_train = ds_type == 'train'  # Whether the dataset is of type train.
    augment_sample = parser.augment_sample
    if is_train:  # Creating the augmentation transform.
        if augment_sample:
            # create a separate augmentation per job since we always update aug_data.aug_seed
            aug_data = GetAugData(image_size)
            aug_data.aug_seed = range_start
            aug_data.augment = True
    # divide the job into several smaller parts and run them sequentially
    ranges = np.arange(range_start, range_stop, parser.job_chunk_size)
    if ranges[-1] != range_stop:
        ranges = ranges.tolist()
        ranges.append(range_stop)
    rel_id = 0
    cur_samples_dir = os.path.join(storage_dir, ds_type)  # Making the path.
    if not os.path.exists(cur_samples_dir):  # creating the train/test/val paths is needed.
        os.makedirs(cur_samples_dir)
    for k in range(len(ranges) - 1):  # Splitting into consecutive jobs.
        range_start = ranges[k]
        range_stop = ranges[k + 1]
        print('%s: job %d. processing: %s-%d-%d' % (
        datetime.datetime.now(), job_id, ds_type, range_start, range_stop - 1))

        print('%s: storing in: %s' % (datetime.datetime.now(), cur_samples_dir))
        sys.stdout.flush()
        for samid in range(range_start, range_stop):
            # Generating the samples.
            sample = gen_sample(parser, samid, ds_type, aug_data, dataloader, examples[rel_id])
            if sample is None:
                continue
            # Stores the samples.
            store_sample_disk(parser,sample, cur_samples_dir)
            rel_id += 1
    print('%s: Done' % (datetime.datetime.now()))

def Split_data_into_jobs_and_generate_samples(parser, raw_data_set, examples, storage_dir, ds_type):
    njobs = parser.nthreads
    job_chunk_size = parser.job_chunk_size
    cur_nexamples = len(examples)
    # each 'job' processes several chunks. Each chunk is of 'storage_batch_size' samples
    cur_njobs = min(njobs, np.ceil(cur_nexamples / job_chunk_size).astype(int))  # The needed number of jobs.
    local_multiprocess = njobs > 1
    ranges = np.linspace(0, cur_nexamples, cur_njobs + 1).astype(int)
    # in case there are fewer ranges than jobs
    ranges = np.unique(ranges)
    all_args = []
    jobs_range = range(len(ranges) - 1)
    cur_samples_dir = os.path.join(storage_dir, ds_type)  # Making the path.
    if not os.path.exists(cur_samples_dir):  # creating the train/test/val paths is needed.
        os.makedirs(cur_samples_dir)
    # Iterating for each job and generate the needed number of samples.
    for job_id in jobs_range:
        range_start = ranges[job_id]
        range_stop = ranges[job_id + 1]
        # Preparing the arguments for the generation.
        args = ( parser, raw_data_set, job_id, range_start, range_stop, examples[range_start:range_stop], storage_dir, ds_type)
        all_args.append(args)
        if not local_multiprocess:
            gen_samples(*args)  # Calling the generation function.
    if local_multiprocess:
        with Pool(cur_njobs) as process:
            process.starmap(gen_samples, all_args)  # Calling the generation function.

def Get_label_ordered(infos:list)->np.array:
    """
    Args:
        infos: The information about all characters.

    Returns: list of all characters arranged in order.
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
    Args:
        infos: The information about the sample.
        nclasses: The number of classes.

    Returns: The label_existence flag, telling for each entry whether the character is in the image.
    """
    label = np.zeros(nclasses) # Initially all zeros.
    for info in infos: # for each character in the image.
        label[info.label] = 1 # Set 1 in the label.
    label_existence = np.array(label)
    return label_existence

def pause_image(fig=None) -> None:
    """
    Pauses the image until a button is presses.
    """
    plt.draw()
    plt.show(block=False)
    if fig is None:
        fig = plt.gcf()
    fig.waitforbuttonpress()

def AddCharacterToExistingImage(DataLoader:DataSet, image:np.array, char:CharInfo)->tuple:
    """
    Function Adding a character to a given image.
    Args:
        DataLoader: The raw data loader.
        image: The image we desire to add a character to.
        char: The character we desire to add.

    Returns: The new image and the info about the character.
    """
    label= char.label # The label of the character.
    label_id = char.label_id # The label -d.
    img_id =DataLoader.num_examples_per_character * label + label_id  # The index in the data-loader.
    im, _ = DataLoader[img_id] # The character image.
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
    part[rng] = mask[rng] * im[rng] + (1 - mask[rng]) * part[rng] # Change the part, yielding changing the original image.
    edge_to_the_right = char.edge_to_the_right
    info = CharInfo(label, label_id,  im, stx, endx, sty, endy, edge_to_the_right)
    return image, info # Return the image, the character info.

def Get_label_task(example:ExampleClass, infos:list,label_ordered:np.array,nclasses:int,keypoints:list)->tuple:
    """
    Args:
        example: The sample example.
        infos: The information about all characters.
        label_ordered: The label_all.
        nclasses: The number of classes.
        keypoints:

    Returns: The label_task, the flag.
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
    keypoint = keypoints[c]
    return label_task,flag,keypoint

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
    avail_ratio = valid_pairs.sum() / (len(valid_classes) *  (len(valid_classes) - 1)) # The number of available pairs.
    exclude_percentage = (100 * (1 - avail_ratio)) # The percentage.
    print('Excluding %d strings, %f percentage of pairs' % (ntest_strings, exclude_percentage))
    return valid_pairs,test_chars_list # return the valid pairs, the test_characters.

def Get_sample_chars(parser, prng:random, valid_pairs:np.array,ds_type:bool,valid_classes:list, ntest_strings:int,test_chars_list:list)->list:
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
    num_characters_per_sample = parser.num_characters_per_sample
    sample_chars = []
    is_test = ds_type =='test'
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

def create_examples_per_sample(parser,prng,ds_type,examples,sample_chars,chars,adj_types):
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
    single_feat_to_generate = parser.single_feat_to_generate
    ncharacters_per_image = parser.num_characters_per_sample
    is_test = ds_type == 'test'
    is_val = ds_type == 'val'
    ngenerate = parser.ngenerate
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
            example = ExampleClass(sample_chars,query_part_id,adj_type,chars )
            examples.append(example)

def Get_valid_classes(use_only_valid_classes,nclasses): #Only for emnist.
    """
    :param parser: 
    :param nclasses: 
    :return: 
    """
   # use_only_valid_classes = parser.use_only_valid_classes  # False #CHANGE IN EMNIST TO TRUE
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

def Make_data_dir(parser, store_folder,language_list):
    """
    :param parser: 
    :param store_folder: 
    :param language_list: 
    :return: 
    """
    base_storage_dir = '%d_' % (parser.nchars_per_row * parser.num_rows_in_the_image)
    base_storage_dir += 'extended_testing_' + str(language_list[0])
    store_dir = store_folder
    base_samples_dir = os.path.join(store_dir, base_storage_dir)
    if not os.path.exists(base_samples_dir):
        os.makedirs(base_samples_dir, exist_ok=True)
    storage_dir = base_samples_dir
    conf_data_fname = os.path.join(storage_dir, 'MetaData')
    return conf_data_fname, storage_dir

def Save_script_if_needed(storage_dir):
    """
    Args:
        storage_dir:

    Returns:
    """
    code_folder_path = os.path.dirname(os.path.realpath(__file__))
    storage_dir = os.path.join(storage_dir, 'code')
    if not os.path.exists(storage_dir):
     shutil.copytree(code_folder_path, storage_dir, copy_function=shutil.copy)
    print("Done saving the source code")