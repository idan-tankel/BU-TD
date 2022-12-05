import argparse
import os.path
from pathlib import Path

from Create_dataset_classes import UnifiedDataSetType, DsType

from torch.utils.data import Dataset

def Get_parser(ds_type:DsType = DsType.Emnist , num_cols=6, num_rows=1,language_list = [0]):
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cols', default=num_cols, type=int,
                        help='The number of characters in the image')

    parser.add_argument('--num_rows', default=num_rows,
                        type=int, help='The number of queries to create for the same image')

    parser.add_argument('--ds_type', default=ds_type,
                        type=DsType,choices=list(DsType), help='The Data_Creation-set type')

    Ds_obj = UnifiedDataSetType(ds_type=ds_type, num_cols=num_cols, num_rows=num_rows,language_list = language_list)
    parser.add_argument('--path_data_raw_for_omniglot',
                        default=os.path.join(Path(__file__).parents[3], 'Data_Creation/Omniglot/RAW/omniglot-py/Unified'), type=str,
                        help='The Raw Data_Creation path')
    parser.add_argument('--store_folder', default=os.path.join(Path(__file__).parents[3],
                         'Data_Creation/') + '{}/samples/'.format(str(ds_type)),   type=str, help='The storing path')
    parser.add_argument('--job_chunk_size', default=1000, type=int, help='The number of samples each jobs processes')
    parser.add_argument('--folder_split', default=True, type=bool, help='Whether to split the folder into parts.')
    parser.add_argument('--folder_size', default=1000, type=int, help=' The folder size')
    parser.add_argument('--augment_sample', default=True, type=bool, help='Whether to augment the sample')
    parser.add_argument('--letter_size', default=Ds_obj.ds_obj.letter_size, type=int, help='The basic letter size')
    parser.add_argument('--nthreads', default=1, type=int, help='The number of threads in each job')
    parser.add_argument('--nsamples_train', default=Ds_obj.ds_obj.nsamples_train,
                        type=int, help='The number of samples in the train set')
    parser.add_argument('--nsamples_test', default=Ds_obj.ds_obj.nsamples_test,
                        type=int, help='The number of samples in the test set')
    parser.add_argument('--nsamples_val', default=Ds_obj.ds_obj.nsamples_val, type=int,
                        help='The number of samples in the val set')
    parser.add_argument('--create_CG_test', default=Ds_obj.ds_obj.create_CG_test, type=bool,
                        help='Whether to create the combinatorial generalization set')
    parser.add_argument('--use_only_valid_classes', default=Ds_obj.ds_obj.use_only_valid_classes,
                        type=bool, help='Whether to use only specific classes')
  #  parser.add_argument('--nclasses',default=Ds_obj.ds_obj.raw_data_set.nclasses,type = Dataset, help = 'The raw images dataset')
    parser.add_argument('--valid_directions', default=[(-1,0),(1,0)], type=int,
                        help='Number of directions to create from')
    parser.add_argument('--image_size', default=Ds_obj.ds_obj.image_size, type=list, help='The image size')
    parser.add_argument('--ngenerate', default=Ds_obj.ds_obj.ngenerate, type=int,
                        help='The number of queries to create for the same image')
    parser.add_argument('--num_characters_per_sample', default=num_cols *
                                                               num_rows, type=int,
                        help='The number of queries to create for the same image')
    parser.add_argument('--num_strings_for_CG', default=1, type=int,
                        help='The number of samples strings for the combinatorial test.')
    parser.add_argument('--minscale', default=Ds_obj.ds_obj.min_scale, type=float,
                        help='The minimal character scale')
    parser.add_argument('--maxscale', default=Ds_obj.ds_obj.max_scale, type=float,
                        help='The maximal character scale')
    parser.add_argument('--minshift', default=Ds_obj.ds_obj.min_shift,
                        type=float, help='The minimal shift')
    parser.add_argument('--maxshift', default=Ds_obj.ds_obj.max_shift,
                        type=float, help='The maximal shift')
    return parser.parse_args()
