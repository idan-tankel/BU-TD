from Create_dataset_funcs import create_samples_for_all_data_set_types_and_save_meta_dat_and_code_script
from Create_dataset_classes import  DsType
from parser import Get_parser
from Raw_data import  DataSet

def main(ds_type, language_list:list)->None:
    # Getting the option parser.
    parser = Get_parser(ds_type)
    raw_data_set = DataSet(parser,data_dir = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/' + ds_type.ds_name, dataset = ds_type,raw_data_source=parser.path_data_raw_for_omniglot,language_list = language_list) # Getting the raw data.
    create_samples_for_all_data_set_types_and_save_meta_dat_and_code_script(parser,raw_data_set, ds_type,language_list=language_list)

def main_Omniglot():
    list = [8,9,11,12,13,141,15,18,20,23,24,27,28,29,32,33,34]
    for i in list:
        main(ds_type=DsType("omniglot"), language_list=[i])


def main_FashionEmnist():
    main(ds_type = DsType("fashionmnist"), language_list = [0])

def main_mnist():
    main(ds_type=DsType("emnist"), language_list = [0])

def main_kmnist():
    main(ds_type = DsType("kmnist"), language_list = [0])

def main_SVHN():
    main(ds_type=DsType("SVHN"), language_list=[0])

main_Omniglot()


