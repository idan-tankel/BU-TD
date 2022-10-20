import argparse

def Get_parser(ds_type, nchars_per_row =6, num_rows_in_the_image = 1):
<<<<<<< HEAD
    # TODO - MAKE ALSO CLASS HERE.
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_data_raw_for_omniglot', default='/home/sverkip/data/BU-TD/yonathan/Recognicion/data/omniglot/RAW', type=str, help='The Raw data path')
    parser.add_argument('--store_folder', default='/home/sverkip/data/BU-TD/yonathan/Recognicion/data/'+ds_type.from_enum_to_str()+'/samples/', type=str,help='The storing path')
=======
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_data_raw_for_omniglot', default='/home/sverkip/data/BU-TD/yonathan/Recognicion/data/omniglot/RAW', type=str, help='The Raw data path')
    parser.add_argument('--store_folder', default='/home/sverkip/data/BU-TD/yonathan/Recognicion/data/'+ds_type.from_enum_to_str()+'/samples_new/', type=str,help='The storing path')
>>>>>>> 315b11ac3016dc72662fd8ca96881ae68c5cda6d
    parser.add_argument('--job_chunk_size', default=1000, type=int, help='The number of samples each jobs processes')
    parser.add_argument('--folder_split', default=True, type=bool, help='Whether to split the folder into parts.')
    parser.add_argument('--folder_size', default=1000, type=int, help=' The folder size')
    parser.add_argument('--augment_sample', default = True, type=bool, help='Whether to augment the sample')
    parser.add_argument('--letter_size', default=28, type=int, help='The basic letter size')
    parser.add_argument('--nthreads', default = 10, type=int, help='The number of threads in the job')
    parser.add_argument('--nchars_per_row', default = nchars_per_row, type=int, help='The number of characters in the image')
<<<<<<< HEAD
    parser.add_argument('--nsamples_train', default = 10, type=int, help='The number of samples in the train set')
    parser.add_argument('--nsamples_test', default = 40, type=int, help='The number of samples in the test set')
    parser.add_argument('--nsamples_val', default = 40, type=int, help='The number of samples in the val set')
    parser.add_argument('--generalize', default = False, type=bool, help='Whether to create the combinatorial generalization set')
    parser.add_argument('--use_only_valid_classes', default = False, type=bool, help='Whether to use only specific classes')
=======
    parser.add_argument('--nsamples_train', default = 20000, type=int, help='The number of samples in the train set')
    parser.add_argument('--nsamples_test', default = 2000, type=int, help='The number of samples in the test set')
    parser.add_argument('--nsamples_val', default = 2000, type=int, help='The number of samples in the val set')
    parser.add_argument('--generalize', default = True, type=bool, help='Whether to create the combinatorial generalization set')
    parser.add_argument('--use_only_valid_classes', default = True, type=bool, help='Whether to use only specific classes')
>>>>>>> 315b11ac3016dc72662fd8ca96881ae68c5cda6d
    parser.add_argument('--ndirections', default = 2, type=int, help='Number of directions to create from')
    parser.add_argument('--image_size', default = [112, 224], type=list, help='The image size')
    parser.add_argument('--create_all_directions', default=False, type=bool,   help='Whether to create all directions')
    parser.add_argument('--ngenerate', default = 5, type=int, help='The number of queries to create for the same image')
    parser.add_argument('--num_characters_per_sample', default = nchars_per_row * num_rows_in_the_image , type=int, help='The number of queries to create for the same image')
    parser.add_argument('--single_feat_to_generate', default=False, type=bool, help='Whether to create multiple queries about the same sample')
    parser.add_argument('--ntest_strings', default = 1, type=int, help='The number of samples strings for the combinatorial test.')
    parser.add_argument('--num_rows_in_the_image', default = num_rows_in_the_image, type=int,  help = 'The number of queries to create for the same image')
<<<<<<< HEAD
    parser.add_argument('--minscale', default = 0.8, type=float,  help = 'The minimal character scale')
    parser.add_argument('--maxscale', default=1.1, type=float, help = 'The maximal character scale')
    parser.add_argument('--minshift', default=0.0, type=float, help = 'The minimal shift')
    parser.add_argument('--maxshift', default=0.0, type=float, help = 'The maximal shift')
=======
    parser.add_argument('--minscale', default = 1.5, type=float,  help = 'The minimal character scale')
    parser.add_argument('--maxscale', default=2.0, type=float, help = 'The maximal character scale')
    parser.add_argument('--minshift', default=1.0, type=float, help = 'The minimal shift')
    parser.add_argument('--maxshift', default=0, type=float, help = 'The maximal shift')
>>>>>>> 315b11ac3016dc72662fd8ca96881ae68c5cda6d
    return parser.parse_args()