import sys
import datetime
# for copying the generating script
import __main__ as mainmod
import shutil
from torchvision import transforms
from multiprocessing import Pool
from Create_dataset_utils import *
from parser import *
from Raw_data_loaders import *

# TODO-assert nclasses is the correct one.


def gen_sample(parser: argparse, sample_id: int, is_train: bool, aug_data: transforms, dataloader: DataSet, example: ExampleClass, augment_sample: bool) -> Sample:
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
    image = 0 * np.ones(parser.image_size, dtype=np.float32)
    infos = []  # Stores all the information about the characters.
    for char in example.chars:  # Iterate over each chosen character.
        # Adding the character to the image.
        image, info = AddCharacterToExistingImage(dataloader, image, char)
        infos.append(info)  # Adding to the info about the characters.
    # Making label_existence flag.
    label_existence = Get_label_existence(infos, dataloader.nclasses)
    # the characters in order as seen in the image
    label_ordered = Get_label_ordered(infos)
    # instruction and task label
    label_task, flag = Get_label_task(
        example, infos, label_ordered, dataloader.nclasses)
    # even for grayscale images, store them as 3 channels RGB like
    if image.shape[0] == 1:
        image = np.concatenate((image, image, image), axis=0)
    # Making RGB.
    image = image * 255
    image = image.astype(np.uint8)
    # Doing data augmentation
    if is_train and augment_sample:
        # augment
        data_augment = DataAugmentClass(image, label_existence, aug_data)
        image = data_augment.get_batch_base()
    # Storing the needed information about the sample.
    sample = Sample(infos, image, sample_id, label_existence,
                    label_ordered, example.query_part_id, label_task, flag, is_train)
    return sample  # Returning the sample we are going to store.


def gen_samples(parser: argparse, dataloader: DataSet, job_id: int, range_start: int, range_stop: int, examples: list, storage_dir: str, ds_type: str, augment_sample: bool) -> None:
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
    for k in range(len(ranges) - 1):  # Splitting into consecutive jobs.
        range_start = ranges[k]
        range_stop = ranges[k + 1]
        print('%s: job %d. processing: %s-%d-%d' %
              (datetime.datetime.now(), job_id, ds_type, range_start, range_stop - 1))
        # Making the path.
        cur_samples_dir = os.path.join(storage_dir, ds_type)
        # creating the train/test/val paths is needed.
        if not os.path.exists(cur_samples_dir):
            os.makedirs(cur_samples_dir)
        print('%s: storing in: %s' %
              (datetime.datetime.now(), cur_samples_dir))
        sys.stdout.flush()
        for samid in range(range_start, range_stop):
            # Generating the samples.
            sample = gen_sample(parser, samid, is_train, aug_data,
                                dataloader, examples[rel_id], augment_sample)
            if sample is None:
                continue
            # Stores the samples.
            store_sample_disk(sample, cur_samples_dir,
                              parser.folder_split, parser.folder_size)
            rel_id += 1
    print('%s: Done' % (datetime.datetime.now()))


def main(language_list: list) -> None:
    """
    main _summary_

    Args:
        language_list (list): _description_
    """    
    # Getting the option parser.
    parser = Get_parser()
    # Getting the raw data.
    raw_data_set = DataSet(data_dir='../data', dataset='emnist',
                           raw_data_source=parser.path_data_raw, language_list=[49])
    parser.image_size = (raw_data_set.nchannels, *parser.image_size)
    njobs = parser.threads  # The number of threads.
    # The number of rows in the image.
    num_rows_in_the_image = parser.num_rows_in_the_image
    # The number of characters in the row.
    obj_per_row = parser.nchars_per_row
    # The number of characters in the image.
    num_chars_per_image = parser.nchars_per_row * parser.num_rows_in_the_image
    # The number of test samples we desire to create.
    nsamples_test = parser.nsamples_test
    # The number of train samples we desire to create.
    nsamples_train = parser.nsamples_train
    # The number of validation samples we desire to create.
    nsamples_val = parser.nsamples_val
    # The number of queries to create for each sample.
    ngenerate = parser.ngenerate
    # The number of classes in the dataset.
    nclasses = raw_data_set.nclasses
    parser.letter_size = raw_data_set.letter_size
    # Whether to create single query per sample.
    single_feat_to_generate = parser.single_feat_to_generate
    # The valid classes, relevant for mnist.
    valid_classes = Get_valid_classes(parser, nclasses)
    # The number of directions we want to query about.
    ndirections = parser.ndirections
    # Whether to create the combinatorial generalization dataset.
    generalize = parser.generalize
    # The number of samples each job should handle.
    job_chunk_size = parser.job_chunk_size
    augment_sample = parser.augment_sample
    # Use multiprocessing on this machine
    local_multiprocess = njobs > 1
    image_ids = set()
    test_chars_list = []
    valid_pairs = None
    ntest_strings = parser.ntest_strings

    # each 'job' processes several chunks. Each chunk is of 'storage_batch_size' samples
    # Get the storage dir for the data and for the conf file.
    conf_data_fname, storage_dir = Get_data_dir(
        parser, parser.store_folder, language_list)
    if parser.create_all_directions:  # Creating the possible tasks
        avail_adj_types = range(ndirections)
    else:
        avail_adj_types = [0]
    ds_types = ['test', 'train']  # The dataset types.
    nexamples_vec = [nsamples_test, nsamples_train]
    if generalize:  # if we want also the CG dataset we add its name to the ds_type and the number of needed samples.
        nexamples_vec.append(nsamples_val)
        ds_types.append('val')
    else:
        nsamples_val = 0
    if generalize:  # If we want the CG dataset, we choose the excluded combinations and pears.
        valid_pairs, test_chars_list = Get_valid_pairs_for_the_combinatorial_test(
            parser, nclasses, valid_classes, num_chars_per_image)
    # Iterating over all dataset types, and its number of desired number of samples.
    for k, (ds_type, cur_nexamples) in enumerate(zip(ds_types, nexamples_vec)):
        prng = np.random.RandomState(k)
        is_train = ds_type == 'train'
        is_val = ds_type == 'val'
        is_test = ds_type == 'test'
        examples = []
        for i in range(cur_nexamples):  # Iterating over the number of samples.
            if generalize:  # If generalize we get the possible pairs we can choose from.
                sample_chars = Get_sample_chars(
                    prng, valid_pairs, is_test, valid_classes, num_chars_per_image, ntest_strings, test_chars_list)
            else:
                # Otherwise we choose from all the valid classes, without replacement the desired number of characters.
                sample_chars = prng.choice(
                    valid_classes, num_chars_per_image, replace=False)
            image_id = []
            label_ids = []
            # For each character, sample an id in the number of characters with the same label.
            for _ in sample_chars:
                # Choose a possible label_id.
                label_id = prng.randint(
                    0, raw_data_set.num_examples_per_character)
                image_id.append(label_id)
                label_ids.append(label_id)
            image_id_hash = str(image_id)
            if image_id_hash in image_ids:
                continue
            image_ids.add(image_id_hash)
            # place the chars on the image
            chars = []  # The augmented characters.
            # For each chosen character, we augment it and transform it.
            for samplei in range(num_chars_per_image):
                char = CharacterTransforms(
                    parser, prng, label_ids, samplei, sample_chars)
                chars.append(char)
            print(i)
            # TODO - ASSERT WHAT WE WANT HERE.
            # For the two directions case this should change.
            # in 6_extended train_iterate_all_directions = False
            # After we chose the characters we choose a direction and a query id.
            adj_types = [prng.choice(avail_adj_types)]
            # generate a single or multiple examples for each generated configuration
       #     adj_types=[0]
            create_examples_per_sample(examples, sample_chars, chars, prng, adj_types,
                                       num_chars_per_image, single_feat_to_generate, is_test, is_val, ngenerate)
        cur_nexamples = len(examples)
        if is_train:  # Update the new number of samples.
            nsamples_train = cur_nexamples
        elif is_test:
            nsamples_test = cur_nexamples
        else:
            nsamples_val = cur_nexamples

        # print the number of sampled examples.
        print('total of %d examples' % cur_nexamples)
        # divide all the examples across several jobs. Each job generates samples from examples
        # The needed number of jobs.
        cur_njobs = min(njobs, np.ceil(
            cur_nexamples / job_chunk_size).astype(int))
        ranges = np.linspace(0, cur_nexamples, cur_njobs + 1).astype(int)
        # in case there are fewer ranges than jobs
        ranges = np.unique(ranges)
        all_args = []
        jobs_range = range(len(ranges) - 1)
        # Iterating for each job and generate the needed number of samples.
        for job_id in jobs_range:
            range_start = ranges[job_id]
            range_stop = ranges[job_id + 1]
            # Preparing the arguments for the generation.
            args = (parser, raw_data_set, job_id, range_start, range_stop,
                    examples[range_start:range_stop], storage_dir, ds_type, augment_sample)
            all_args.append(args)
            if not local_multiprocess:
                gen_samples(*args)  # Calling the generation function.
        if local_multiprocess:
            with Pool(cur_njobs) as process:
                # Calling the generation function.
                process.starmap(gen_samples, all_args)

    print('done')  # Done creating and storing the samples.
    # store the dataset's properties.
    with open(conf_data_fname, "wb") as new_data_file:
        pickle.dump((nsamples_train, nsamples_test, nsamples_val, nclasses,  parser.letter_size, parser.image_size,
                    num_rows_in_the_image, obj_per_row, num_chars_per_image, ndirections, valid_classes), new_data_file)
    # TBD - SAVE ALSO THE UTILS.
    # copy the generating script
    script_fname = mainmod.__file__
    shutil.copy(script_fname, storage_dir)


# %%
if __name__ == "__main__":
  #   tasks = [[27,5]]
    tasks = [[17]]
    for task in tasks:
        main(task)
