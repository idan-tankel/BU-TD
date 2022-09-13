from torchvision import transforms
from Create_dataset_funcs import gen_samples, Get_valid_classes, Make_data_dir, Create_raw_examples, Split_data_into_jobs_and_generate_samples, Save_script_if_needed
from Create_dataset_classes import MetaData
from parser import Get_parser
from Raw_data import Get_raw_data, DataSet, DsType
import os
import pickle
import numpy as np

def main(ds_type, language_list:list)->None:
    # Getting the option parser.
    parser = Get_parser(ds_type)
    raw_data_set = DataSet(data_dir = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/'+ds_type,dataset = ds_type,raw_data_source=parser.path_data_raw_for_omniglot,language_list = language_list) # Getting the raw data.
    parser.image_size = (raw_data_set.nchannels,*parser.image_size) # The number of threads.
    nsamples_test = parser.nsamples_test           # The number of test samples we desire to create.
    nsamples_train = parser.nsamples_train         # The number of train samples we desire to create.
    nsamples_val = parser.nsamples_val             # The number of validation samples we desire to create.             # The number of queries to create for each sample.
    nclasses = raw_data_set.nclasses            # The number of classes in the dataset.
    parser.letter_size = raw_data_set.letter_size
    valid_classes = Get_valid_classes(parser.use_only_valid_classes, nclasses)  # The valid classes, relevant for mnist.
    generalize = parser.generalize   # Whether to create the combinatorial generalization dataset.
    image_ids = set()
    conf_data_fname, storage_dir = Make_data_dir(parser, parser.store_folder,language_list) # Get the storage dir for the data and for the conf file.
    ds_types = ['test', 'train'] # The dataset types.
    nexamples_vec = [nsamples_test, nsamples_train]
    if generalize: # if we want also the CG dataset we add its name to the ds_type and the number of needed samples.
        nexamples_vec.append(nsamples_val)
        ds_types.append('val')
        valid_pairs, test_chars_list = Get_valid_pairs_for_the_combinatorial_test(parser, nclasses, valid_classes)
    else:
        nsamples_val = 0
        valid_pairs, test_chars_list = None, []
    # Iterating over all dataset types, and its number of desired number of samples.
    for k, (ds_type, cur_nexamples) in enumerate(zip(ds_types, nexamples_vec)):
        examples = Create_raw_examples(parser, image_ids, k, ds_type, cur_nexamples, valid_pairs, valid_classes,test_chars_list, raw_data_set.num_examples_per_character)
        cur_nexamples = len(examples)
        if ds_type == 'train': # Update the new number of samples.
             nsamples_train = cur_nexamples
        elif ds_type == 'test':
            nsamples_test = cur_nexamples
        elif nsamples_val > 0 :
            nsamples_val = cur_nexamples
        print('total of %d examples' % cur_nexamples) # print the number of sampled examples.
        # divide all the examples across several jobs. Each job generates samples from examples
        Split_data_into_jobs_and_generate_samples(parser,raw_data_set, examples, storage_dir, ds_type)

    print('done') # Done creating and storing the samples.
    # store the dataset's properties.
    with open(conf_data_fname, "wb") as new_data_file:
        struct = MetaData(parser, nsamples_train, nsamples_test, nsamples_val, valid_classes)
        pickle.dump(struct, new_data_file)
    Save_script_if_needed(storage_dir)

def main_Omniglot():
   main(ds_type=DsType.Omniglot, language_list= [9])

def main_FashionEmnist():
    main(ds_type=DsType.FashionMnist, language_list = [0])

def main_mnist():
    main(ds_type=DsType.Emnist, language_list = [0])

main_FashionEmnist()
#main_Omniglot()
#main_Emnist()
