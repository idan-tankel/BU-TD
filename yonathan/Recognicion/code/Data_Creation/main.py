import os.path
from pathlib import Path
from typing import Union

from Create_dataset_classes import DsType,UnifiedDataSetType
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
    parser = Get_parser(ds_type = ds_type, num_rows=num_rows_in_the_image, num_cols=num_chars_per_row)
    Data_path = os.path.join(Path(__file__).parents[2], 'Data_Creation')
    Raw_data_path = os.path.join(Data_path, str(ds_type))

   # raw_data_set = Raw_data_set(parser, raw_data_dir=Raw_data_path,
   #                             raw_data_source=parser.path_data_raw_for_omniglot,
    #                            language_list=language_list)  # Getting the raw Data_Creation.
    # Iterating over all dataset types, and its number of desired number of samples.
    raw_data = UnifiedDataSetType(ds_type=ds_type, num_cols=num_chars_per_row, num_rows=num_rows_in_the_image,language_list=language_list).ds_obj.raw_data_set
    nsamples_per_data_type_dict = create_dataset(parser,raw_data, ds_type,  language_list)
    print(
        'Done creating and storing the samples, we are left only with saving the meta Data_Creation and the code script.')
    # Done creating and storing the samples.
    Save_meta_data_and_code_script(parser, ds_type, nsamples_per_data_type_dict,
                                   language_list)  # Saving the MataData and code script.
    print('Done saving the source code and the meta Data_Creation!')


if __name__ == '__main__':
    # main(ds_type=DsType.Omniglot,language_list=[27, 5, 42, 18, 33])
    main(ds_type=DsType.Emnist, num_chars_per_row=4, num_rows_in_the_image=4)
   # main(ds_type=DsType.Emnist, num_chars_per_row=6, num_rows_in_the_image=1)
#  for i in range(50):
#   main(ds_type=DsType.Emnist,language_list=[i])
