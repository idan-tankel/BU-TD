from parser import Get_parser
import Create_dataset_classes
import Create_dataset_funcs
from Raw_data import DataSet
import yaml
from types import SimpleNamespace
import git


def main(ds_type: DataSet, language_list=None, nchars_per_row=6, num_rows_in_the_image=1) -> None:
    """
    Args:
        parser: The dataset options.
        raw_data_set: The raw dataset.
        ds_type: The dataset type.
        language_list: The language list.

    """
    parser = Get_parser(ds_type, nchars_per_row=nchars_per_row,
                        num_rows_in_the_image=num_rows_in_the_image)
    parser = get_create_config()
    raw_data_set = DataSet(parser, data_dir='/home/idanta/data/' + ds_type.from_enum_to_str(),  dataset=ds_type,
                           raw_data_source=parser.path_data_raw,  language_list=language_list)  # Getting the raw data.
    # Iterating over all dataset types, and its number of desired number of samples.
    nsamples_per_data_type_dict = Create_dataset_funcs.create_samples(
        parser, ds_type, raw_data_set, language_list)
    # Done creating and storing the samples.
    print('Done creating and storaging the samples, we are left only with saving the meta data and the code script.')
    Create_dataset_funcs.Save_meta_data_and_code_script(
        parser, ds_type, nsamples_per_data_type_dict,  language_list)
    print('Done saving the source code and the meta data!')


def main_Omniglot(lists):
    for list in lists:
        main(ds_type=Create_dataset_classes.DsType.Omniglot, language_list=list)


def main_FashionEmnist(nchars_per_row=6, num_rows_in_the_image=1):
    main(ds_type=Create_dataset_classes.DsType.Fashionmist, nchars_per_row=nchars_per_row,
         num_rows_in_the_image=num_rows_in_the_image)


def main_emnist(nchars_per_row=6, num_rows_in_the_image=1):
    main(ds_type=Create_dataset_classes.DsType.Emnist, nchars_per_row=nchars_per_row,
         num_rows_in_the_image=num_rows_in_the_image)


def main_kmnist():
    main(ds_type=Create_dataset_classes.DsType.Kmnist)


def get_create_config():
    git_repo = git.Repo(__file__, search_parent_directories=True)
    git_root = git_repo.working_dir
    full_path = f"{git_root}/new_revision/create_dataset/create_config.yaml"
    with open(full_path, 'r') as stream:
        config_as_dict = yaml.safe_load(stream)
    config_as_namespace = SimpleNamespace(**config_as_dict)
    config_as_namespace.num_characters_per_sample = config_as_namespace.nchars_per_row * \
        config_as_namespace.num_rows_in_the_image
    return config_as_namespace


if __name__ == '__main__':
    main_emnist(1, 1)


