import os
import argparse
import string
import yaml
from types import SimpleNamespace


def get_parser(nchars_per_row=6, num_rows_in_the_image=1):
    """
    Get_parser _summary_

    #TODO change this to config.yaml or config.ini file
    Args:
        nchars_per_row (int, optional): Numbers of rows per incoming data . Defaults to 6.
        num_rows_in_the_image (int, optional): Numbers of raws in each image. Defaults to 1.

    Returns:
        `argparse`: an object holding up all the arguments for running (coming from terminal)
    """    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='emnist',help='name of the dataset to use')
    parser.add_argument('--path_data_raw', default='../data',
                        type=str, help='The Raw data path')
    parser.add_argument('--store_folder', default='../data',
                        type=str, help='The storing path')
    parser.add_argument('--job_chunk_size', default=1000, type=int,
                        help='The number of samples each jobs processes')
    parser.add_argument('--folder_split', default=True, type=bool,
                        help='Whether to split the folder into parts.')
    parser.add_argument('--folder_size', default=1000,
                        type=int, help=' The folder size')
    parser.add_argument('--augment_data', default=True,
                        type=bool, help='Whether to augment the data')
    parser.add_argument('--letter_size', default=28,
                        type=int, help='The basic letter size')
    parser.add_argument('--threads', default=10, type=int,
                        help='The number of threads in the job')
    parser.add_argument('--nchars_per_row', default=nchars_per_row,
                        type=int, help='The number of characters in the image')
    parser.add_argument('--nsamples_train', default=10000,
                        type=int, help='The number of samples in the train set')
    parser.add_argument('--nsamples_test', default=2000,
                        type=int, help='The number of samples in the test set')
    parser.add_argument('--nsamples_val', default=2000,
                        type=int, help='The number of samples in the val set')
    parser.add_argument('--generalize', default=True, type=bool,
                        help='Whether to create the combinatorial generalization set')
    parser.add_argument('--use_only_valid_classes', default=True,
                        type=bool, help='Whether to use only specific classes')
    parser.add_argument('--ndirections', default=2, type=int,
                        help='Number of directions to create from')
    parser.add_argument(
        '--image_size', default=[112, 224], type=list, help='The image size')
    parser.add_argument('--create_all_directions', default=False,
                        type=bool,   help='Whether to create all directions')
    parser.add_argument('--ngenerate', default=5, type=int,
                        help='The number of queries to create for the same image')
    parser.add_argument('--num_rows_in_the_image', default=num_rows_in_the_image,
                        type=int, help='The number of rows in the image')
    parser.add_argument('--single_feat_to_generate', default=False, type=bool,
                        help='Whether to create multiple queries about the same sample')
    parser.add_argument('--ntest_strings', default=1, type=int,
                        help='The number of samples strings for the combinatorial test.')
    parser.add_argument('--sample_nchars', default=nchars_per_row * num_rows_in_the_image,
                        type=int,   help='The number of characters in each image')
    return parser.parse_args()

def get_config():
    """
    Get_config _summary_

    Returns:
        `namespace`: an object holding up all the arguments for running (coming from config file)
    """

    real_path = os.path.realpath(__file__)
    dir_path = os.path.dirname(real_path)
    full_path = f"{dir_path}/Configs/create_config.yaml"
    with open(full_path, 'r') as stream:
        config_as_dict = yaml.safe_load(stream)
    config_as_namespace= SimpleNamespace(**config_as_dict)
    config_as_namespace.sample_nchars = config_as_namespace.nchars_per_row * config_as_namespace.num_rows_in_the_image
    return config_as_namespace
