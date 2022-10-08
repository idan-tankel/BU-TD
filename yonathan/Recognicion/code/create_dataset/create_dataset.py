from Create_dataset_funcs import main
from Create_dataset_classes import  DsType

def main_Omniglot(lists):
    for list in lists:
        main(ds_type = DsType.Omniglot, language_list = list)

def main_FashionEmnist():
    main(ds_type = DsType.Fashionmist)

def main_emnist():
    main(ds_type=DsType.Emnist)

def main_kmnist():
    main(ds_type = DsType.Kmnist)

main_Omniglot([[35]])
#main_Omniglot([[14],[39]])
#main_emnist()


