"""
The main function, creating the dataset according to the desired type.
"""
from typing import Union

from src.Create_dataset_classes import DsType, UnifiedDataSetType
from src.Create_dataset_funcs import create_dataset, Save_meta_data_and_code_script
from src.data_creation_parser import Get_parser


def main(language_list: Union[list, None] = None, num_chars_per_row: int = 6,
         num_rows_in_the_image: int = 1) -> None:
    """
    The main function calling 'create_dataset'.
    Creating the image matrix, created by concatenating some images in a matrix order.
    Args:
        language_list: The language list.
        num_chars_per_row: Number of characters per row.
        num_rows_in_the_image: Number of rows per in the image.

    """
    ds_type: DsType = DsType.Omniglot
    parser = Get_parser(ds_type=ds_type, num_rows=num_rows_in_the_image, num_cols=num_chars_per_row,
                        language_list=language_list)
    # Iterating over all dataset types, and its number of desired number of inputs.
    raw_data = UnifiedDataSetType(ds_type=ds_type, num_cols=num_chars_per_row, num_rows=num_rows_in_the_image,
                                  language_list=language_list).ds_obj.raw_data_set
    nsamples_per_data_type_dict = create_dataset(parser, raw_data, ds_type, language_list)
    print(
        'Done creating and storing the inputs, we are left only with saving the meta data and the code script.')
    # Done creating and storing the inputs.
    Save_meta_data_and_code_script(parser, ds_type, nsamples_per_data_type_dict,
                                   language_list)  # Saving the MataData and code script.
    print('Done saving the source code and the meta data!')


if __name__ == '__main__':
    # Initial tasks.
    main(language_list=[48, 47, 46, 45, 44])
    # New tasks.
    main(language_list=[49])
    main(language_list=[43])
    main(language_list=[42])
    main(language_list=[41])
    main(language_list=[40])
