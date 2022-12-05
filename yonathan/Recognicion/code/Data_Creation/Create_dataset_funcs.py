import argparse
import datetime
import os
import pickle
import random
import shutil
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image
from Create_dataset_classes import DsType, Sample, CharInfo, DataAugmentClass, MetaData


def store_sample_disk(parser: argparse, sample: Sample, store_dir: str) -> None:
    """
    Storing the sample on the disk.
    Args:
        parser: The Data_Creation parser.
        sample: The sample we desire to save.
        store_dir: The directory we store into.
    """

    samples_dir = store_dir
    i = sample.sample_id
    if parser.folder_split:
        samples_dir = os.path.join(store_dir,
                                   '%d' % (i // parser.folder_size))  # The inner path, based on the sample id.
        if not os.path.exists(samples_dir):  # Make the path to the sample.
            os.makedirs(samples_dir, exist_ok=True)
    img = sample.image  # The image.
    img_fname = os.path.join(samples_dir, '%d_img.jpg' % i)  # The image directory.
    c = Image.fromarray(img.transpose(1, 2, 0))  # Convert to Image format.
    c.save(img_fname)  # Saving the image.
    data_fname = os.path.join(samples_dir, '%d_raw.pkl' % i)  # The labels' directory.
    del sample.image
    with open(data_fname, "wb") as new_data_file:  # Dumping the supervision labels in the pickle file.
        pickle.dump(sample, new_data_file)


def AddCharToImage(image: np.array, char: CharInfo) -> np.ndarray:
    """
    Adding a character to the matrix image.
    Args:
        image: The image we desire to add a character to.
        char: The character we desire to add.
    Returns: The new image.
    """
    img = char.img  # The image we desire to plant.
    stx = char.stx  # The beginning coordinate on the x-axis.
    end_x = char.end_x  # The end coordinate on the x-axis.
    sty = char.sty  # The beginning coordinate on the y-axis.
    end_y = char.end_y  # The end coordinate on the y-axis.
    # This is a view into the image and when it changes the image also changes.
    image[:, sty:end_y, stx:end_x] = img  # planting the character.
    return image  # Return the image.


def create_image_matrix(parser: argparse, ds_type: str, sample: Sample) -> Sample:
    """
    Creates a single sample including image, label_task, label_all, label_existence, query_index.
    Args:
        parser: The option parser.
        ds_type: The Data_Creation set type.
        sample: The sample containing the selected characters.

    Returns: A Sample to store on the disk.
    """

    augment_sample = parser.augment_sample
    is_train = ds_type == 'train'
    for char in sample.chars:  # Iterate over each chosen character.
        sample.image = AddCharToImage(sample.image, char)  # Adding the character to the image.
    # Even for grayscale images, store them as 3 channels RGB like
    if sample.image.shape[0] == 1:  # If gray-scale.
        sample.image = np.concatenate((sample.image, sample.image, sample.image), axis=0)
    # Making RGB.
    sample.image = sample.image * 255
    sample.image = sample.image.astype(np.uint8)
    # Doing Data_Creation augmentation if needed.
    if is_train and augment_sample:
        # augment
        data_augment = DataAugmentClass(sample.sample_id)
        sample.image = data_augment(sample.image)
    return sample


def gen_samples(parser: argparse, job_id: int, range_start: int, range_stop: int,
                samples: list[Sample],
                storage_dir: str, ds_type: str) -> None:
    """
    Generates and stored samples, by calling to create_sample and store_sample_disk.
    Args:
        parser: The option parser.
        job_id: The job id.
        range_start: The range start in the job.
        range_stop: The range stop of the job.
        samples: The chosen samples.
        storage_dir: The storage directory
        ds_type: The Data_Creation-set type options: 'train', 'test, 'val'.

    """
    num_folders = int(np.ceil((range_stop - range_start) / parser.job_chunk_size) + 1)
    ranges = np.linspace(range_start, range_stop, num=num_folders, dtype=int, endpoint=True)
    cur_samples_dir = os.path.join(storage_dir, ds_type)  # Making the path.
    num_samples_created_so_far = 0
    if not os.path.exists(cur_samples_dir):  # creating the train/test/val paths if needed.
        os.makedirs(cur_samples_dir)
    for k in range(len(ranges) - 1):  # Splitting into consecutive jobs.
        range_start = ranges[k]  # The current start-point.
        range_stop = ranges[k + 1]  # The current end-point.
        print('%s: job %d. processing: %s-%d-%d' % (
            datetime.datetime.now(), job_id, ds_type, range_start, range_stop - 1))
        print('%s: storing in: %s' % (datetime.datetime.now(), cur_samples_dir))
        sys.stdout.flush()
        for real_id, sample_id in enumerate(range(range_start, range_stop)):
            # Generating the samples.
            sample = create_image_matrix(parser, ds_type, samples[real_id + num_samples_created_so_far])
            # Stores the samples.
            store_sample_disk(parser, sample, cur_samples_dir)
        num_samples_created_so_far += parser.job_chunk_size
    print('%s: Done' % (datetime.datetime.now()))


def Get_valid_pairs_for_the_combinatorial_test(parser: argparse, nclasses: int, valid_classes: np.ndarray) -> \
        tuple[np.ndarray, list[np.ndarray]]:
    """
    Exclude part of the training Data_Creation. Test set is from the train distribution.
    Validation is only the excluded Data_Creation (combinatorial generalization).

    Args:
        parser: The option parser.
        nclasses: The number of classes.
        valid_classes: The valid classes.

    Returns: The valid pairs, we can sample from for the train, test samples, and the CG sequences.
    """
    num_chars_to_sample = parser.num_characters_per_sample
    num_strings_for_CG = parser.num_strings_for_CG
    # generate once the same test strings
    np.random.seed(0)
    valid_pairs = np.zeros((nclasses, nclasses))
    for i in valid_classes:
        for j in valid_classes:
            if j != i:
                valid_pairs[i, j] = 1
    test_chars_list = []
    # it is enough to validate only right-of pairs even when we query for left of as it is symmetric
    for _ in range(num_strings_for_CG):
        test_chars = np.random.choice(valid_classes, num_chars_to_sample, replace=False)  # Choose a sequence.
        print('test chars:', test_chars)
        test_chars_list.append(test_chars)
        test_chars = test_chars.reshape(-1, parser.num_cols)
        for row in test_chars:
            for pi in range(parser.num_cols - 1):
                cur_char = row[pi]  # The current character.
                adj_char = row[pi + 1]  # The adjacent character.
                # Now in each sample will be no pair which is pair in the sampled characters.
                valid_pairs[cur_char, adj_char] = 0

    print(f'Excluding {num_strings_for_CG} strings')
    return valid_pairs, test_chars_list  # return the valid pairs, the test_characters.


def Choose_Chars(parser: argparse, prng: random, valid_pairs: np.array, ds_type: str, valid_classes: np.ndarray,
                 test_chars_list: list[np.ndarray]) -> np.ndarray:
    """
    Returns a valid choice sequence of characters.
    If the sample is from the validation dataset then the sample will be one of the test_chars_list,
    Otherwise it is built according to the CG rule to have valid CG test.
    Args:
        parser: The dataset parser.
        prng: The random generator.
        valid_pairs: The valid pairs to sample from.
        ds_type: The dataset type.
        valid_classes: The valid classes.
        test_chars_list: The sequences for the CG test.

    Returns: The sampled characters.

    """
    num_characters_per_sample = parser.num_characters_per_sample  # The number of chars
    num_strings_for_CG = parser.num_strings_for_CG  # The number of CG strings
    sample_chars = []
    is_val = ds_type == 'val'  # If the sample belongs to the CG test.
    if not is_val:  # If we are not in the CG test.
        found = False
        while not found:  # While we didn't find any valid sequence, we continue.
            sample_chars = []
            cur_char = prng.choice(valid_classes, 1)[0]  # Choose the first character.
            sample_chars.append(cur_char)  # Add the character to the characters.
            for _ in range(num_characters_per_sample - 1):
                cur_char_adj_types = valid_pairs[cur_char].nonzero()[0]  # Valid neighbor.
                cur_char_adj_types = np.setdiff1d(cur_char_adj_types,
                                                  sample_chars)  # We want each character to be at most once.
                if len(cur_char_adj_types) > 0:  # If there is a valid neighbor.
                    cur_adj = prng.choice(cur_char_adj_types, 1)[0]  # Choose valid neighbor.
                    sample_chars.append(cur_adj)  # Add to the sample list.
                    # If we have for enough for sample, we break and return the sample.
                    if len(sample_chars) == num_characters_per_sample:
                        found = True
                    cur_char = cur_adj  # Update the current character to be the neighbor.
                else:  # There is no valid neighbor, we need to resample.
                    break
        # We are in the CG test.
    else:
        test_chars_idx = prng.randint(num_strings_for_CG)  # Sample CG sequence index.
        sample_chars = test_chars_list[test_chars_idx]  # Choose the sequence.
    return sample_chars


def Create_several_samples_per_sequence(parser: argparse, prng: random, ds_type: str, samples: list[Sample],
                                        chars: list[CharInfo], sample_id: int):
    """
    Given sampled sequence of characters, generate several samples, and add to the samples list.
    Args:
        parser: The Data_Creation parser.
        prng: The random generator.
        ds_type: The Data_Creation-set type.
        samples: The samples list.
        chars: The list of all information about all characters
        sample_id: The sample id.
    """
    num_characters_per_image = parser.num_characters_per_sample  # The number of characters in the image.ss
    is_train = ds_type == 'train'  # Whether we are in the train samples.
    ngenerate = parser.ngenerate if is_train else 1  # For train, we create ngenerate queries, for others single one.
    valid_directions = parser.valid_directions  # The valid directions we sample from.
    valid_queries = range(num_characters_per_image)  # All queries are valid.

    query_part_ids = prng.choice(valid_queries, ngenerate, replace=False)  # Choose ngenerate positions
    for query_id, query_part_id in enumerate(query_part_ids):  # For each position, sample direction and create sample.
        direction_id = prng.choice(len(valid_directions))  # Sample direction id.
        adj_type = valid_directions[direction_id]  # The direction.
        sample = Sample(parser, query_part_id, adj_type, chars,
                        ngenerate * sample_id + query_id)  # Create sample.
        samples.append(sample)  # Add to the samples list.


def Get_valid_classes_for_emnist_only(data_set_type: DsType, use_only_valid_classes: bool,
                                      nclasses: int) -> np.array:  # Only for emnist.
    """
    To avoid confusion of the model, we omit some classes from the original number of classes.
    Args:
        data_set_type: The dataset name. If emnist we omit some classes o.w. we take them all.
        use_only_valid_classes: Whether to use only valid classes.
        nclasses: The number of classes.

    Returns: The valid classes.

    """
    # If we are in emnist, and we want to omit confusing classes we do it.
    if use_only_valid_classes and data_set_type is DsType.Emnist:
        # remove some characters which are very similar to other characters
        # ['O', 'F', 'q', 'L', 't', 'g', 'C', 'S', 'I', 'B', 'Y', 'n', 'b', 'X', 'r', 'H', 'P', 'G']
        invalid = [24, 15, 44, 21, 46, 41, 12, 28, 18, 11, 34, 43, 37, 33, 45, 17, 25, 16]
        all_classes = np.arange(0, nclasses)
        valid_classes = np.setdiff1d(all_classes, invalid)
    # Otherwise we choose all classes.
    else:
        valid_classes = np.arange(0, nclasses)
    return valid_classes


def Make_data_dir(parser: argparse, ds_type: DsType, language_list: list) -> tuple:
    """
    Making the Data_Creation-dir for the samples.
    Args:
        parser: The Data_Creation parser.
        ds_type: The Data_Creation set type.
        language_list: The language list.

    Returns: The sample path, the meta-Data_Creation path.
    """
    folder_name = '(%d,%d)_' % (parser.num_rows, parser.num_cols)  # The Data_Creation set specification.
    if ds_type is DsType.Omniglot:
        folder_name += 'data_set_matrix' + str(language_list[0])
    else:
        folder_name += 'data_set_matrix_test_changes2'
    Samples_dir = os.path.join(parser.store_folder, folder_name)  # The path we store into.
    if not os.path.exists(Samples_dir):  # Making the samples dir.
        os.makedirs(Samples_dir, exist_ok=True)
    Meta_data_fname = os.path.join(Samples_dir, 'MetaData')  # Make the metadata dir.
    return Samples_dir, Meta_data_fname


def Generate_raw_samples(parser: argparse, raw_dataset, image_ids: set, data_set_index: int, ds_type: str,
                         cur_nsamples: int,
                         valid_pairs: np.array, valid_classes: np.array, test_chars_list: list[np.ndarray]
                         ) -> list[Sample]:
    """
    Given the valid pairs, valid classes, we generate raw samples for each Data_Creation-set type.
    Here we just choose the character sequence.
    Args:
        parser: The dataset parser.
        raw_dataset: The raw image dataset.
        image_ids: The image ids.
        data_set_index: The seed for each dataset.
        ds_type: The Data_Creation-set type.
        cur_nsamples: The number of characters to generate.
        valid_pairs: The valid pairs.
        valid_classes: The valid classes/
        test_chars_list: The test sequence for the CG(combinatorial generalization) test.

    Returns: List of the generated samples.

    """
    prng = np.random.RandomState(data_set_index)  # Random Generator depending on the Data_Creation set id.
    num_chars_per_image = parser.num_characters_per_sample  # The number of characters we need to choose.
    samples = []  # Initialize the samples list.
    num_samples_created_so_far = 0
    # Iterating over the number of samples.
    while num_samples_created_so_far < cur_nsamples:
        # Sampling a sequence.
        if parser.create_CG_test:  # If generalize we get the possible pairs we can choose from.
            sample_chars = Choose_Chars(parser, prng, valid_pairs, ds_type, valid_classes, test_chars_list)
        else:
            # Otherwise we choose from all the valid classes, without replacement the desired number of characters.
            sample_chars = prng.choice(valid_classes, num_chars_per_image, replace=False)
        image_id_hash = str(sample_chars)
        # To ensure we have each sequence at most one we check whether it's in the image_ids.
        #  if image_id_hash in image_ids:
        #     continue
        image_ids.add(image_id_hash)
        chars = []  # The chosen characters.
        for sample_id in range(num_chars_per_image):
            char = CharInfo(parser, raw_dataset, prng, sample_id, sample_chars)  # Create raw sample.
            chars.append(char)  # Add to the characters list.
        # Create several samples for the chosen sequence.
        Create_several_samples_per_sequence(parser, prng, ds_type, samples, chars,
                                            num_samples_created_so_far)
        num_samples_created_so_far += 1
    return samples


def Save_code_script(storage_dir: str) -> None:
    """
    Saving the code generating the samples.
    Args:
        storage_dir: The path we desire to save the code script.
    """
    code_folder_path = os.path.dirname(os.path.realpath(__file__))
    storage_dir = os.path.join(storage_dir, 'data')
    if not os.path.exists(storage_dir):
        shutil.copytree(code_folder_path, storage_dir, copy_function=shutil.copy)  # Save the code script.


def Save_meta_data_and_code_script(parser: argparse, ds_type: DsType, nsamples_per_data_type_dict: dict,
                                   language_list: list) -> None:
    """
    Saving the metadata and the code script.
    Args:
        parser: The dataset options.
        ds_type: The Data_Creation set type.
        nsamples_per_data_type_dict: The number of samples per dataset.
        language_list: The language list.

    """
    storage_dir, meta_data_fname = Make_data_dir(parser, ds_type, language_list)
    with open(meta_data_fname, "wb") as new_data_file:
        # Creating a struct containing the parser and number of samples per Data_Creation set type dict.
        struct = MetaData(parser,
                          nsamples_per_data_type_dict)
        pickle.dump(struct, new_data_file)  # Dump to the memory.
    Save_code_script(storage_dir)  # Save the code script


def Split_samples_into_jobs_and_generate(parser: argparse, samples: list[Sample],
                                         storage_dir: str, ds_type: str) -> None:
    """
    After the sequence are chosen, we generate samples by splitting into parallel jobs and
    generate the samples including all supervision.
    Args:
        parser: The dataset parser.
        samples: The sampled sequence.
        storage_dir: The storage directory.
        ds_type: The dataset type.

    """
    num_jobs = parser.nthreads  # The number of threads.
    job_chunk_size = parser.job_chunk_size  # The number of samples per job.
    num_samples = len(samples)  # The number of samples
    # each 'job' processes several chunks. Each chunk is of 'storage_batch_size' samples.
    cur_num_jobs = min(num_jobs, np.ceil(num_samples / job_chunk_size).astype(int))  # The needed number of jobs.
    use_multiprocess = num_jobs > 1  # Whether to use multiprocessing.
    ranges = np.linspace(0, num_samples, cur_num_jobs + 1, endpoint=True).astype(int)  # Split the range into jobs.
    # in case there are fewer ranges than jobs
    ranges = np.unique(ranges)
    all_args = []
    jobs_range = range(len(ranges) - 1)
    cur_samples_dir = os.path.join(storage_dir, ds_type)  # Making the path.
    if not os.path.exists(cur_samples_dir):  # creating the train/test/val paths if needed.
        os.makedirs(cur_samples_dir)
    # Iterating for each job and generate the needed number of samples.
    for job_id in jobs_range:
        range_start = ranges[job_id]  # Start range.
        range_stop = ranges[job_id + 1]  # Stop range.
        # Preparing the arguments for the generation.
        args = (parser, job_id, range_start, range_stop, samples[range_start:range_stop], storage_dir, ds_type)
        all_args.append(args)
        # For not using multiprocessing.
        if not use_multiprocess:
            gen_samples(*args)  # Calling the generation function in a sequential manner.
    # For using multiprocessing.
    if use_multiprocess:
        with Pool(cur_num_jobs) as process:
            process.starmap(gen_samples, all_args)  # Calling the generation function in a parallel manner.


def create_dataset(parser: argparse, raw_dataset, ds_type: DsType, language_list: list) -> dict:
    """
    The main function choosing sequences, then generating samples, Stores them and saves the MetaData.
    Args:
        parser: The dataset options.
        raw_dataset: The raw dataset images.
        ds_type: All Data_Creation-set types.
        language_list: The language list for omniglot only.

    Returns: Generating samples and returning a dictionary assigning for each dataset the actual
    number of generated samples.

    """

    nclasses = raw_dataset.nclasses  # The number of classes in the dataset.
    valid_classes = Get_valid_classes_for_emnist_only(ds_type, parser.use_only_valid_classes,
                                                      nclasses)  # The valid classes, relevant for mnist.
    generalize = parser.create_CG_test  # Whether to create the combinatorial generalization dataset.
    image_ids = set()  # The image_ids, needed for avoiding choosing the same sequence twice.
    # The number of queries to create for each sample:
    nsamples_test = parser.nsamples_test  # The number of test samples we desire to create.
    nsamples_train = parser.nsamples_train  # The number of train samples we desire to create.
    nsamples_val = parser.nsamples_val  # The number of validation samples we desire to create.
    storage_dir, _ = Make_data_dir(parser, ds_type, language_list)
    # Get the storage dir for the Data_Creation and for the conf file.
    ds_types = ['test', 'train']  # The dataset types.
    nsamples_vec = [nsamples_test, nsamples_train]
    if generalize:  # if we want also the CG dataset we add its name to the ds_type and the number of needed samples.
        nsamples_vec.append(nsamples_val)
        ds_types.append('val')
        # Generating valid pairs we can sample from.
        valid_pairs, CG_chars_list = Get_valid_pairs_for_the_combinatorial_test(parser, nclasses,
                                                                                valid_classes)
    else:
        valid_pairs, CG_chars_list = None, []

    num_samples_per_data_type_dict = {}
    # Iterating over all dataset types and generating raw samples for each and then generating samples.
    for k, (ds_type, cur_nsamples) in enumerate(zip(ds_types, nsamples_vec)):
        samples = Generate_raw_samples(parser, raw_dataset, image_ids, k, ds_type, cur_nsamples, valid_pairs,
                                       valid_classes,
                                       CG_chars_list)
        cur_nsamples = len(samples)
        num_samples_per_data_type_dict[ds_type] = cur_nsamples
        print('total of %d samples' % cur_nsamples)  # print the number of samples.
        # divide all the samples across several jobs. Each job generates different samples.
        Split_samples_into_jobs_and_generate(parser, samples, storage_dir, ds_type)
    return num_samples_per_data_type_dict
