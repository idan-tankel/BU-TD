from typing import Union

from Create_dataset_classes import DsType, UnifiedDataSetType
from Create_dataset_funcs import create_dataset, Save_meta_data_and_code_script
from parser import Get_parser


def main(ds_type: DsType = DsType.Emnist, language_list: Union[list, None] = None, num_chars_per_row: int = 6,
         num_rows_in_the_image: int = 1) -> None:
    """
    The main function calling 'create_dataset'.
    Args:
        ds_type: The dataset option e.g. emnist, Fashiomnist.
        language_list: The language list.
        num_chars_per_row: Number of characters per row.
        num_rows_in_the_image: Number of rows per in the image.

    """
    parser = Get_parser(ds_type=ds_type, num_rows=num_rows_in_the_image, num_cols=num_chars_per_row)
    # Iterating over all dataset types, and its number of desired number of samples.
    raw_data = UnifiedDataSetType(ds_type=ds_type, num_cols=num_chars_per_row, num_rows=num_rows_in_the_image,
                                  language_list=language_list).ds_obj.raw_data_set
    nsamples_per_data_type_dict = create_dataset(parser, raw_data, ds_type, language_list)
    print(
        'Done creating and storing the samples, we are left only with saving the meta data and the code script.')
    # Done creating and storing the samples.
    Save_meta_data_and_code_script(parser, ds_type, nsamples_per_data_type_dict,
                                   language_list)  # Saving the MataData and code script.
    print('Done saving the source code and the meta data!')


if __name__ == '__main__':
    # main_fashion(ds_type=DsType.Omniglot,language_list=[27, 5, 42, 18, 33])
    main(ds_type=DsType.Emnist, num_chars_per_row=4, num_rows_in_the_image=4)
# main_fashion(ds_type=DsType.Emnist, num_chars_per_row=6, num_rows_in_the_image=1)
# for i in [43, 24]:
#  main_fashion(ds_type=DsType.Omniglot,language_list=[i])
