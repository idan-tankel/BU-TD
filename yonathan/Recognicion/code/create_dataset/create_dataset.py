from parser import Get_parser
from Create_dataset_classes import DsType
from Create_dataset_funcs import create_samples, Save_meta_data_and_code_script
from Raw_data import DataSet


def main(ds_type:DataSet, language_list = None, nchars_per_row =6, num_rows_in_the_image = 1 ) -> None:
    """
    Args:
        parser: The dataset options.
        raw_data_set: The raw dataset.
        ds_type: The dataset type.
        language_list: The language list.

    """
    parser = Get_parser(ds_type, nchars_per_row = nchars_per_row, num_rows_in_the_image = num_rows_in_the_image)
    raw_data_set = DataSet(parser, data_dir='/home/idanta/data' + ds_type.from_enum_to_str(),  dataset=ds_type, raw_data_source = parser.path_data_raw_for_omniglot,  language_list=language_list)  # Getting the raw data.
    # Iterating over all dataset types, and its number of desired number of samples.
    nsamples_per_data_type_dict = create_samples(parser, ds_type, raw_data_set, language_list )
    print('Done creating and storaging the samples, we are left only with saving the meta data and the code script.')  # Done creating and storing the samples.
    Save_meta_data_and_code_script(parser, ds_type, nsamples_per_data_type_dict,  language_list)
    print('Done saving the source code and the meta data!')

def main_Omniglot(lists):
    for list in lists:
        main(ds_type = DsType.Omniglot, language_list = list)

def main_FashionEmnist(nchars_per_row =6, num_rows_in_the_image = 1):
    main(ds_type = DsType.Fashionmist, nchars_per_row = nchars_per_row, num_rows_in_the_image = num_rows_in_the_image)

def main_emnist(nchars_per_row =6, num_rows_in_the_image = 1):
    main(ds_type = DsType.Emnist, nchars_per_row = nchars_per_row, num_rows_in_the_image = num_rows_in_the_image)

def main_kmnist():
    main(ds_type = DsType.Kmnist)

main_emnist(6,4)

#main_Omniglot([[35]])
#main_Omniglot([[14],[39]])
#main_emnist()


