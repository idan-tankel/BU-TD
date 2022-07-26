import sys
import numpy as np
import datetime
# for copying the generating script
import __main__ as mainmod
import shutil
from torchvision import transforms
from multiprocessing import Pool
from Create_dataset_utils import *
from parser import *

#TODO-assert nclasses is the correct one.
def gen_sample(sample_id:int, is_train:bool, aug_data:transforms, OmniglotLoader:OmniglotDataLoader, nclasses:int, CHAR_SIZE:int, IMAGE_SIZE:list, example:example_class, augment_sample,num_examples_per_character:int)->Sample:
    """
    #
    Creates a single sample including image, label_task, label_all, label_existence, query_index.
    #
    :param sample_id: The sample id in all samples.
    :param is_train: If the sample is for training.
    :param aug_data:
    :param OmniglotLoader:
    :param nclasses: The augmentation transform.
    :param CHAR_SIZE: The character size.
    :param IMAGE_SIZE: The desired image size.
    :param example: The character example we sampled.
    :param augment_sample: Whether to augment the sample or not.
    :param num_examples_per_character: The num of examples per language
    """
    # start by creating the image background(all black)
    image = 0 * np.ones(IMAGE_SIZE, dtype=np.float32)
    infos = [] #Stores all the information about the characters.
    for char in example.chars: # Iterate for each chosen character.
     (image,info) = AddCharacterToExistingImage(OmniglotLoader,image ,char,CHAR_SIZE,num_examples_per_character) # Adding the character to the image.
     infos.append(info) # Adding to the info about the characters.
    #Making label_existence flag.
    label_existence = Get_label_existence(infos,nclasses) 
    # the characters in order as seen in the image
    label_ordered = Get_label_ordered(infos)
    # instruction and task label
    label_task, flag, query_index = Get_label_task(example, infos,label_ordered,nclasses)
    # even for grayscale images, store them as 3 channels RGB like
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
    # Making RGB.
    image = 255 * np.concatenate((image, image, image), axis=2)
    image = image.astype(np.uint8)
    # Doing data augmentation
    if is_train and augment_sample:
        # augment
        Data_augment = DataAugmentClass(image, label_existence, aug_data)
        image = Data_augment.get_batch_base()
    # Storing the needed information about the sample.
    sample = Sample(infos, image, sample_id, label_existence, label_ordered, query_index, label_task, flag, is_train)
    return sample # Returning the sample we are going to store.

def gen_samples(parser,Omniglot_loader, job_id, range_start, range_stop, examples, storage_dir,ds_type, nclasses, job_chunk_size, augment_sample, folder_split, folder_size)->None:
    """
    #
    Generates and stored samples, by calling to create_sample and store_sample_disk_pytorch
    #
    :param parser: The creation parser.
    :param Omniglot_loader: The data loader.
    :param job_id: The job id.
    :param range_start: The range start. 
    :param range_stop: The range stop.
    :param examples: The examples.
    :param storage_dir: The directory to store in.
    :param ds_type: The ds_type.
    :param nclasses: The number of classes.
    :param job_chunk_size: The number of samples each job processes.
    :param augment_sample: Whether to augment the sample.
    :param folder_split: Whether to split the folder.
    :param folder_size: The folder size.
    """
    CHAR_SIZE = parser.CHAR_SIZE # The character size.
    IMAGE_SIZE = parser.IMAGE_SIZE # The image size.
    aug_data = None # The augmentation transform.
    is_train = ds_type == 'train' # Whether the dataset is of type train.
    if is_train: # Creating the augmentation transform.
        if augment_sample:
            # create a separate augmentation per job since we always update aug_data.aug_seed
            aug_data = get_aug_data(IMAGE_SIZE)
            aug_data.aug_seed = range_start
            aug_data.augment = True
    # divide the job into several smaller parts and run them sequentially
    ranges = np.arange(range_start, range_stop, job_chunk_size)
    if ranges[-1] != range_stop:
        ranges = ranges.tolist()
        ranges.append(range_stop)
    rel_id = 0
    for k in range(len(ranges) - 1): # Splitting into consecutive jobs.
        range_start = ranges[k]
        range_stop = ranges[k + 1]
        print('%s: job %d. processing: %s-%d-%d' % (datetime.datetime.now(), job_id, ds_type, range_start, range_stop - 1))
        cur_samples_dir = os.path.join(storage_dir, ds_type)  # Making the path.
        if not os.path.exists(cur_samples_dir): # creating the train/test/val paths is needed.
            os.makedirs(cur_samples_dir)
        print('%s: storing in: %s' % (datetime.datetime.now(), cur_samples_dir))
        sys.stdout.flush()
        for samid in range(range_start, range_stop):
            # Generating the samples.
            sample = gen_sample(samid, is_train, aug_data, Omniglot_loader, nclasses, CHAR_SIZE, IMAGE_SIZE, examples[rel_id], augment_sample,Omniglot_loader.num_examples_per_character)
            if sample is None:
                continue
            # Stores the samples.
            store_sample_disk(sample, cur_samples_dir, folder_split, folder_size)
            rel_id += 1
    print('%s: Done' % (datetime.datetime.now()))



from yonathan.emnist_dataset import *

def main(language_list:list)->None:
  #  cmd_args = parser.parse_args()
    Omniglot_loader = EmnistLoader(data_dir='/home/sverkip/data/Create_dataset_adapting_to_all_datasets/data') # Getting the raw data.
    parser = Get_parser()


  #  dictionary = create_dict(parser.path_data_raw)  # The dictionary assigning for each language its number of characters.
  #  dictionary = dict(sorted(dictionary.items(), key=lambda item: item[1]))
  #  Omniglot_loader = EmnistLoader(parser.store_folder)
    # Getting the option parser.
    njobs = parser.threads # The number of threads.
    dictionary = create_dict(parser.path_data_raw) # The dictionary assigning for each language its number of characters.
    num_rows_in_the_image   = parser.num_rows_in_the_image      # The number of rows in the image.
    obj_per_row = parser.nchars_per_row            # The number of characters in the row.
    num_chars_per_image = parser.nchars_per_row * parser.num_rows_in_the_image       # The number of characters in the image.
    nsamples_test = parser.nsamples_test           # The number of test samples we desire to create.
    nsamples_train = parser.nsamples_train         # The number of train samples we desire to create.
    nsamples_val = parser.nsamples_val             # The number of validation samples we desire to create.
    ngenerate = parser.ngenerate                   # The number of queries to create for each sample.
    nclasses = Omniglot_loader.nclasses            # The number of classes in the dataset.
    single_feat_to_generate = parser.single_feat_to_generate # Whether to create single query per sample.
    valid_classes = Get_valid_classes(parser, nclasses)  # The valid classes, relevant for mnist.
    ndirections = parser.ndirections               # The number of directions we want to query about.
    generalize = parser.generalize   # Whether to create the combinatorial generalization dataset.
    job_chunk_size = parser.job_chunk_size # The number of samples each job should handle.
    augment_sample = parser.augment_sample
    # Use multiprocessing on this machine
    local_multiprocess = njobs > 1 # NOT SURE-ASK LIAV.
    image_ids = set()

    # each 'job' processes several chunks. Each chunk is of 'storage_batch_size' samples
    conf_data_fname, storage_dir = Get_data_dir(parser, parser.store_folder,language_list) # Get the storage dir for the data and for the conf file.
    if parser.create_all_directions: # Creating the possible tasks
     avail_adj_types = range(ndirections)
    else:
     avail_adj_types = [0]

    ds_types = ['test', 'train'] # The dataset types.
    nexamples_vec = [nsamples_test, nsamples_train]
    if generalize: # if we want also the CG dataset we add its name to the ds_type and the number of needed samples.
        nexamples_vec.append(nsamples_val)
        ds_types.append('val')
    else:
        nsamples_val = 0

    if generalize: # If we want the CG dataset, we choose the excluded combinations and pears.
        valid_pairs,test_chars_list = Get_valid_pairs_for_the_combenatorial_test(parser, nclasses, valid_classes, num_chars_per_image, obj_per_row)
        ntest_strings = 1
    # Iterating over all dataset types, and its number of desired number of samples.
    for k, (ds_type, cur_nexamples) in enumerate(zip(ds_types, nexamples_vec)):
        prng = np.random.RandomState(k)
        is_train = ds_type == 'train'
        is_val = ds_type == 'val'
        is_test = ds_type == 'test'
        examples = []
        for i in range(cur_nexamples): # Iterating over the number of samples.
            if generalize: # If generalize we get the possible pairs we can choose from.
                sample_chars = Get_sample_chars(prng, valid_pairs, is_test, valid_classes, num_chars_per_image, ntest_strings, test_chars_list)
            else:
                # Otherwise we choose from all the valid classes, without replacement the desired number of characters.
                sample_chars = prng.choice(valid_classes, num_chars_per_image, replace=False)
            image_id = []
            label_ids = []
            # For each character, sample an id in the number of characters with the same label.
            for _ in sample_chars:
                label_id = prng.randint(0, Omniglot_loader.num_examples_per_character ) # Choose a possible label_id.
                image_id.append(label_id)
                label_ids.append(label_id)
            image_id_hash = str(image_id)
            if image_id_hash in image_ids: # ASK LIAV
                continue
            image_ids.add(image_id_hash)
            # place the chars on the image
            chars = [] # The augmented characters.
            for samplei in range(num_chars_per_image): #For each chosen character, we augment it and transform it.
                char = CharcterTransforms(prng, parser.CHAR_SIZE,label_ids,samplei,sample_chars,num_rows_in_the_image ,obj_per_row,parser.IMAGE_SIZE,num_chars_per_image)
                chars.append(char)
            print(i)
            #TODO - ASSERT WHAT WE WANT HERE.
            # For the two directions case this should change.
            #in 6_extended train_iterate_all_directions = False
            # After we chose the characters we choose a direction and a query id.
            adj_types = [prng.choice(avail_adj_types)]
            # generate a single or multiple examples for each generated configuration
       #     adj_types=[0]
            create_examples_per_sample(examples,sample_chars,chars, prng, adj_types,num_chars_per_image,single_feat_to_generate,is_test,is_val,ngenerate)
        cur_nexamples = len(examples)
        if is_train: # Update the new number of samples.
            nsamples_train = cur_nexamples
        elif is_test:
            nsamples_test = cur_nexamples
        else:
            nsamples_val = cur_nexamples

        print('total of %d examples' % cur_nexamples) # print the number of sampled examples.
        # divide all the examples across several jobs. Each job generates samples from examples
        cur_njobs = min(njobs, np.ceil((cur_nexamples) / job_chunk_size).astype(int)) # The needed number of jobs.
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
            args = (parser, Omniglot_loader, job_id, range_start, range_stop, examples[range_start:range_stop], storage_dir, ds_type, nclasses, job_chunk_size,   augment_sample,  parser.folder_split, parser.folder_size)
            all_args.append(args)
            if not local_multiprocess:
             gen_samples(*args) # Calling the generation function.
        if local_multiprocess:
            with Pool(cur_njobs) as p:
                p.starmap(gen_samples, all_args)#  Calling the generation function.

    print('done') # Done creating and storing the samples.
    # store the dataset's properties.
    with open(conf_data_fname, "wb") as new_data_file:
        pickle.dump((nsamples_train, nsamples_test, nsamples_val, nclasses,  parser.CHAR_SIZE, parser.IMAGE_SIZE, num_rows_in_the_image, obj_per_row, num_chars_per_image,ndirections, valid_classes), new_data_file)
    #TODO -SAVE ALSO THE UTILS.
    # copy the generating script
    script_fname = mainmod.__file__
    shutil.copy(script_fname, storage_dir)
# %%
if __name__ == "__main__":
  #   tasks = [[27,5]]
     tasks = [ [17] ]
     for task in tasks:
      ret = main(task)