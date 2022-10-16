import argparse
import datetime
import os
import pickle
import random
import shutil
import sys
from multiprocessing import Pool
from parser import Get_parser

import numpy as np
import skimage
import skimage.transform as transforms
import torch.utils.data as data
from PIL import Image

from Create_dataset_classes import Sample, ExampleClass, CharInfo, DataAugmentClass, MetaData
from Raw_data import DataSet


def store_sample_disk(parser:argparse,sample:Sample, store_dir:str)->None:
    """
    Storing the sample on the disk.
    Args:
        parser: The data parser.
        sample: The sample we desire to save.
        store_dir: The directory we desire to save into.
    """

    samples_dir =  store_dir
    i = sample.id
    if parser.folder_split:
        samples_dir = os.path.join(store_dir, '%d' % (i // parser.folder_size)) # The inner path, based on the sample id.
        if not os.path.exists(samples_dir):
            os.makedirs(samples_dir, exist_ok=True)
    img = sample.image # Saving the image.
    img_fname = os.path.join(samples_dir, '%d_img.jpg' % i) # The image directory.
    c = Image.fromarray(img.transpose(1,2,0)) # Convert to Image format.
    c.save(img_fname) # Saving the image.
    data_fname = os.path.join(samples_dir, '%d_raw.pkl' % i) # The labels directory.
    del sample.image
    with open(data_fname, "wb") as new_data_file: # Dumping the labels, and the flags in the pickle file.
        pickle.dump(sample, new_data_file)

def Generate_raw_examples(parser:argparse, image_ids:list, k:int, ds_type:str, cur_nexamples:int,valid_pairs:np.array, valid_classes:np.array, test_chars_list:list, data_set:DataSet)->list:
    """
    Given the valid pairs, valid, we generate raw samples for each data-set type.
    Here we just choose the characters and there label_id (A specific character of all characters with the same label).
    Args:
        parser: The dataset parser.
        image_ids: The image ids.
        k: The seed for each dataset.
        ds_type: The data-set type.
        cur_nexamples: The number of characters to generate.
        valid_pairs: The valid pairs.
        valid_classes: The valid classes/
        test_chars_list: The test sequence for the CG(combinatorial generalization) test.
        data_set: The raw dataset.

    Returns: List of the generated examples.

    """
    prng = np.random.RandomState(k)
    num_chars_per_image = parser.num_characters_per_sample
    examples = [] # Initialize the examples list.
    for i in range(cur_nexamples):  # Iterating over the number of samples.
        # Sampling a sequence.
        if parser.generalize:  # If generalize we get the possible pairs we can choose from.
            sample_chars = Get_sample_chars(parser,prng, valid_pairs, ds_type, valid_classes, test_chars_list)
        else:
            # Otherwise we choose from all the valid classes, without replacement the desired number of characters.
            sample_chars = prng.choice(valid_classes, num_chars_per_image, replace=False)
        image_id = []
        label_ids = []
        # For each character, sample an id from the number of characters with the same label.
        for _ in sample_chars:
            label_id = prng.randint(0,data_set.num_examples_per_character)  # Choose a possible label_id.
            image_id.append(label_id)
            label_ids.append(label_id)
        image_id_hash = str(image_id)
        if image_id_hash in image_ids:
            continue
        image_ids.add(image_id_hash)
        chars = []  # The augmented characters.
        for samplei in range(num_chars_per_image):  # For each chosen character, we augment it and transform it.
            char = CharInfo(parser, prng, label_ids, samplei, sample_chars)
            chars.append(char)
        print(i)
        if parser.create_all_directions:  # Creating the possible tasks
            avail_adj_types = range(parser.ndirections)
        else:
            avail_adj_types = [0]
        adj_types = [prng.choice(avail_adj_types)]
        create_examples_per_sample(parser, prng, ds_type, examples, sample_chars, chars, adj_types)
    return examples

def gen_sample(parser: argparse, sample_id: int, ds_type: str, dataloader: DataSet, example: ExampleClass) -> Sample:
    """
    Creates a single sample including image, label_task, label_all, label_existence, query_index.
    Args:
        parser: The option parser.
        sample_id: The sample id.
        ds_type: The data set type.
        aug_data: The data augmentation transform.
        dataloader: The raw data loader.
        example: The example containing the selected characters.

    Returns: A Sample to store on the disk.
    """
    augment_sample = parser.augment_sample
    is_train = ds_type == 'train'
    # start by creating the image background(all black)
    image = 0 * np.ones(parser.image_size, dtype=np.float32)
    for char in example.chars:  # Iterate over each chosen character.
        image = AddCharacterToExistingImage(dataloader, image, char)  # Adding the character to the image.
    # Making label_existence flag.
    label_existence = Get_label_existence(example, dataloader.nclasses)
    # the characters in order as seen in the image
    label_ordered = Get_label_ordered(example)
    # even for grayscale images, store them as 3 channels RGB like
    if image.shape[0] == 1:
        image = np.concatenate((image, image, image), axis=0)
    # Making RGB.
    image = image * 255
    image = image.astype(np.uint8)
    # instruction and task label
    label_task, flag = Get_label_task(example, label_ordered, dataloader)
    # Doing data augmentation if needed.
    if is_train and augment_sample:
        # augment
        data_augment = DataAugmentClass(sample_id)
        image = data_augment(image)
    # Storing the needed information about the sample in the class.
    sample = Sample(image, sample_id, label_existence, label_ordered, example.query_part_id, label_task, flag, is_train)
    return sample  # Returning the sample we are going to store.

def gen_samples(parser: argparse, dataloader: DataSet, job_id: int, range_start: int, range_stop: int, examples: list, storage_dir: str, ds_type: str) -> None:
    """
    Generates and stored samples, by calling to create_sample and store_sample_disk.
    Args:
        parser: The option parser.
        dataloader: The raw data loader.
        job_id: The job id.
        range_start: The range start in the job.
        range_stop: The range stop of the job.
        examples: The chosen examples.
        storage_dir: The storage directory
        ds_type: The data-set type.

    """
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
            sample = gen_sample(parser, samid, ds_type,  dataloader, examples[rel_id])
            if sample is None:
                continue
            # Stores the samples.
            store_sample_disk(parser,sample, cur_samples_dir)
            rel_id += 1
    print('%s: Done' % (datetime.datetime.now()))

def Split_examples_into_jobs_and_generate_samples(parser:argparse, raw_data_set:DataSet, examples:list, storage_dir:str, ds_type:str):
    """
    After the examples are generate, we generate samples by splitting into parallel jobs and generate the samples including all supervision.
    Args:
        parser: The dataset parser.
        raw_data_set: The raw dataset.
        examples: The generated examples.
        storage_dir: The storage directory.
        ds_type: The dataset type.

    """
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

def Get_label_ordered(example:list[ExampleClass])->np.array:
    """
    Generate the label_ordered label.
    Args:
        example: The information about all characters.

    Returns: list of all characters arranged in order.
    """
    row = list()
    rows = [] #All the rows.
    for char in example.chars: # Iterate for each character in the image.
        character = char.label
        row.append(character) # All the characters in the row.
        if char.edge_to_the_right:
            rows.append(row)
            row = list()
    label_ordered = np.array(rows)
    return label_ordered

def Get_label_existence(example:list[ExampleClass], nclasses:int)->np.array:
    """
    Generating the label_existence label.
    Args:
        example: The information about the sample.
        nclasses: The number of classes.

    Returns: The label_existence flag, telling for each entry whether the character is in the image.
    """
    label = np.zeros(nclasses) # Initially all zeros.
    for char in example.chars: # for each character in the image.
        label[char.label] = 1 # Set 1 in the label.
    label_existence = np.array(label)
    return label_existence

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
    img_id = DataLoader.num_examples_per_character * label + label_id  # The index in the data-loader.
    im, _ = DataLoader[img_id] # The character image.
    scale = char.scale
    c = DataLoader.nchannels
    # Scaling character and getting the desired location to plant the character.
    sz = scale * np.array(im.shape[1:])
    sz = sz.astype(np.int)
    h, w = sz
    sz = (c,*sz)
    im = skimage.transform.resize(im, sz, mode='constant') # Apply the transform.
    stx = char.location_x
    endx = stx + w
    sty = char.location_y
    endy = sty + h
    # planting the character.
    # this is a view into the image and when it changes the image also changes
    part = image[:,sty:endy, stx:endx] # The part of the image we plan to plant the character.
    mask = im.copy()
    mask[mask > 0] = 1
    mask[mask < 1] = 0
    rng = mask > 0
    part[rng] = mask[rng] * im[rng] + (1 - mask[rng]) * part[rng] # Change the part, yielding changing the original image.
    return image # Return the image, the character info.

def Get_label_task(example:ExampleClass, label_ordered:np.array,data_loader:data.Dataset)->tuple:
    """
    Args:
        example: The sample example.
        label_ordered: The label_all.
        data_loader: The number of classes.

    Returns: The label_task, the flag.
    """
    query_part_id = example.query_part_id # The index we create about.
    char = example.chars[query_part_id] # The character information in the index.
    character = char.label # The label of the character.
    rows, obj_per_row = label_ordered.shape # The number of rows, the number of characters in evert row.
    adj_type = example.adj_type # The direction we query about.
    edge_class = data_loader. nclasses # The label of the edge class is the number of characters.
    r, c = (label_ordered == character).nonzero() # Find the row and the index of the character.
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
    flag = np.array([adj_type, character]) # The flag.
    return label_task,flag

def Get_valid_pairs_for_the_combinatorial_test(parser:argparse, nclasses:int,valid_classes:list)->tuple:
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

    Returns: The valid pairs and the list of all chosen sequences.
    """
    num_chars_to_sample = parser.nchars_per_row * parser.num_rows_in_the_image
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

def Get_sample_chars(parser, prng:random, valid_pairs:np.array,ds_type:bool,valid_classes:list, test_chars_list:list)->list:
    """
    Returns a valid sample.
    If the sample is from the validation dataset then the sample will be one of the test_chars_list.
    Otherwise it is build according to the CG rules.
    Args:
        parser: The dataset parser.
        prng: The random generator.
        valid_pairs: The valid pairs to sample from.
        ds_type: The dataset type.
        valid_classes: The valid classes.
        test_chars_list: The sequences for the CG test.

    Returns: The sampled characters.

    """
    num_characters_per_sample = parser.num_characters_per_sample
    ntest_strings = parser.ntest_strings
    sample_chars = []
    is_val = ds_type == 'val'
    if not is_val:
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

def create_examples_per_sample(parser:argparse,prng:random,ds_type:str,examples:list,sample_chars:list,chars:list,adj_types:list):
    """
    Adding examples to the examples list.
    Args:
        parser: The data parser.
        prng: The random generator.
        ds_type: The data-set type.
        examples: The examples list.
        sample_chars: The sampled characters list.
        chars: The list of all information about all characters
        adj_types: The sampled directions.
    """
    single_feat_to_generate = parser.single_feat_to_generate
    ncharacters_per_image = parser.num_characters_per_sample
    is_train = ds_type =='train'
    ngenerate = parser.ngenerate
    for adj_type in adj_types:
        valid_queries = range(ncharacters_per_image)
        if single_feat_to_generate or not is_train:
            query_part_ids = [prng.choice(valid_queries)]
        else:
            query_part_ids = prng.choice(valid_queries, ngenerate, replace=False)
        for query_part_id in query_part_ids:
            example = ExampleClass(sample_chars,query_part_id,adj_type,chars )
            examples.append(example)

def Get_valid_classes_for_emnist_only(ds_name, use_only_valid_classes:bool,nclasses:int)->np.array: #Only for emnist.
    """
    To avoid confusion of the model, we omit some classes from the original number of classes.
    Args:
        ds_name: The dataset name. If emnist we omit some classes o.w. we take them all.
        use_only_valid_classes: Whether to use only valid classes.
        nclasses: The number of classes.

    Returns: The valid classes.

    """

    if use_only_valid_classes and ds_name.from_enum_to_str() == 'emnist':
        # remove some characters which are very similar to other characters
        # ['O', 'F', 'q', 'L', 't', 'g', 'C', 'S', 'I', 'B', 'Y', 'n', 'b', 'X', 'r', 'H', 'P', 'G']
        invalid = [24, 15, 44, 21, 46, 41, 12, 28, 18, 11, 34, 43, 37, 33, 45, 17, 25, 16]
        all_classes = np.arange(0, nclasses)
        valid_classes = np.setdiff1d(all_classes, invalid)
    else:
        valid_classes = np.arange(0, nclasses)
    return valid_classes

def Make_data_dir(parser:argparse,ds_type, language_list:list)->tuple:
    """
    Making the data-dir, for the samples.
    Args:
        parser: The data parser.
        store_folder: The folder to store into.
        language_list: The language list.

    Returns: The samples path, the meta-data path.
    """
    base_storage_dir = '%d_' % (parser.num_characters_per_sample)
    if ds_type.from_enum_to_str() == 'omniglot':
     base_storage_dir += 'extended_testing' + str(language_list[0])
    else:
        base_storage_dir += 'extended'
    base_samples_dir = os.path.join(parser.store_folder, base_storage_dir)
    if not os.path.exists(base_samples_dir):
        os.makedirs(base_samples_dir, exist_ok=True)
    Meta_data_fname = os.path.join(base_samples_dir, 'MetaData')
    return base_samples_dir, Meta_data_fname

def create_samples(parser:argparse,ds_type, raw_data_set:DataSet, language_list)->dict:
    """
    Args:
        parser: The dataset options.
        ds_type: All data-set types.
        raw_data_set: The raw dataset.
        language_list: The language list for omniglot only.

    Returns: Generating samples and returning a dictionary assigning for each dataset the actual number of generated samples.

    """

    nclasses = raw_data_set.nclasses  # The number of classes in the dataset.
    parser.letter_size = raw_data_set.letter_size
    valid_classes = Get_valid_classes_for_emnist_only(ds_type, parser.use_only_valid_classes, nclasses)  # The valid classes, relevant for mnist.
    generalize = parser.generalize  # Whether to create the combinatorial generalization dataset.
    image_ids = set()
    nsamples_test = parser.nsamples_test  # The number of test samples we desire to create.
    nsamples_train = parser.nsamples_train  # The number of train samples we desire to create.
    nsamples_val = parser.nsamples_val  # The number of validation samples we desire to create.             # The number of queries to create for each sample.
    storage_dir, _ = Make_data_dir(parser, ds_type, language_list)
    # Get the storage dir for the data and for the conf file.
    ds_types = ['test', 'train']  # The dataset types.
    nexamples_vec = [nsamples_test, nsamples_train]
    if generalize:  # if we want also the CG dataset we add its name to the ds_type and the number of needed samples.
        nexamples_vec.append(nsamples_val)
        ds_types.append('val')
        valid_pairs, CG_chars_list = Get_valid_pairs_for_the_combinatorial_test(parser, nclasses, valid_classes) # Generating valid pairs sampling from.
    else:
        valid_pairs, CG_chars_list = None, []

    num_samples_per_data_type_dict = {}
    # Iterating over all dataset types and generating raw examples for each and then generating samples.
    for k, (ds_type, cur_nexamples) in enumerate(zip(ds_types, nexamples_vec)):
        examples = Generate_raw_examples(parser, image_ids, k, ds_type, cur_nexamples, valid_pairs, valid_classes,CG_chars_list, raw_data_set)
        cur_nexamples = len(examples)
        num_samples_per_data_type_dict[ds_type] = cur_nexamples
        print('total of %d examples' % cur_nexamples) # print the number of sampled examples.
        # divide all the examples across several jobs. Each job generates samples from examples
        Split_examples_into_jobs_and_generate_samples(parser,raw_data_set, examples, storage_dir, ds_type)
    return num_samples_per_data_type_dict

def main( ds_type:DataSet, language_list = None ) -> None:
    """
    Args:
        parser: The dataset options.
        raw_data_set: The raw dataset.
        ds_type: The dataset type.
        language_list: The language list.

    """
    parser = Get_parser(ds_type)
    raw_data_set = DataSet(parser, data_dir='/home/sverkip/data/BU-TD/yonathan/Recognicion/data/' + ds_type.from_enum_to_str(),  dataset=ds_type, raw_data_source = parser.path_data_raw_for_omniglot,  language_list=language_list)  # Getting the raw data.
    # Iterating over all dataset types, and its number of desired number of samples.
    nsamples_per_data_type_dict = create_samples(parser, ds_type, raw_data_set, language_list )
    print('Done creating and storaging the samples, we are left only with saving the meta data and the code script.')  # Done creating and storing the samples.
    Save_meta_data_and_code_script(parser, ds_type, nsamples_per_data_type_dict,  language_list)
    print('Done saving the source code and the meta data!')

def Save_meta_data_and_code_script(parser:argparse,ds_type, nsamples_per_data_type_dict:dict, language_list:list):
    """
    Saving the metadata and the code.
    Args:
        parser: The dataset options.
        nsamples_per_data_type_dict: The dictionary we desire to save.
        valid_classes: The valid classes.
        language_list: The language list.

    """
    storage_dir, meta_data_fname = Make_data_dir(parser,ds_type, language_list)
    with open(meta_data_fname, "wb") as new_data_file:
        struct = MetaData(parser, nsamples_per_data_type_dict)
        pickle.dump(struct, new_data_file)
    Save_script_if_needed(storage_dir)

def Save_script_if_needed(storage_dir:str):
    """
    Saving the code generating the samples.
    Args:
        storage_dir: The path we desire to save the code script.
    """
    code_folder_path = os.path.dirname(os.path.realpath(__file__))
    storage_dir = os.path.join(storage_dir, 'code')
    if not os.path.exists(storage_dir):
     shutil.copytree(code_folder_path, storage_dir, copy_function=shutil.copy)
