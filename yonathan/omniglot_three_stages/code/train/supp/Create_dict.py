import os


def fs(path):
    size = 0
    for _ in os.scandir(path):
        size += 1
    return size


def create_dict(path):
    dict_language = {}
    cnt = 0
    for ele in os.scandir(path):
        path_new = ele
        dict_language[cnt] = fs(path_new)
        cnt += 1
    return dict_language


# %% augmentation

Data_source = '/home/sverkip/data/BU-TD/omniglot/data/omniglot_all_languages'
dictionary = create_dict(Data_source)
# print(dictionary)
print(dict(sorted(dictionary.items(), key=lambda item: item[1])))
print(dictionary[27] + dictionary[5] + dictionary[42])
