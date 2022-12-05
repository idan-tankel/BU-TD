import os
import shutil

from torchvision import datasets


def folder_size(path: str) -> int:
    """
    Returns the number of files in a given folder.
    Args:
        path: Path to a language file.

    Returns: Number of files in the folder
    """
    return len(list(os.scandir(path)))


def create_dict(path: str) -> dict:
    """
    Creates a dictionary assigning for each path in the folder the number of files in it.
    Args:
        path: Path to all raw Omniglot languages.

    Returns: Dictionary of number of characters per language

    """
    dict_language = {}
    for cnt, ele in enumerate(os.scandir(path)):  # for language in Omniglot_raw find the number of characters in it.
        dict_language[ele] = folder_size(ele)  # Find number of characters in the folder.
    return dict_language


def Download_raw_omniglot_data(raw_data_path: str) -> dict:
    """
    Download all Omniglot languages and merge into one folder in 'Unified'.
    Args:
        raw_data_path: The path we want to store into.

    """
    # Raw_data_path = os.path.join(Path(__file__).parents[3], 'Data_Creation/Omniglot/RAW')
    if not os.path.exists(raw_data_path):
        datasets.Omniglot(download=True, root=raw_data_path, background=True)
        datasets.Omniglot(download=True, root=raw_data_path, background=False)
        # Both Data_Creation types.
        list_dir = ['images_background', 'images_evaluation']
        current_folder = os.path.join(raw_data_path, 'omniglot-py')
        content_list = {}
        for index, val in enumerate(list_dir):
            path = os.path.join(current_folder, val)
            content_list[list_dir[index]] = os.listdir(path)

        # folder in which all the content will
        # be merged
        merge_folder = "Unified"
        merge_folder_path = os.path.join(current_folder, merge_folder)
        # create merge_folder if not exists
        if not os.path.exists(merge_folder_path):
            os.mkdir(merge_folder_path)
        # loop through the list of folders
        # Merge all in merge_folder_path.
        for sub_dir in content_list:
            # loop through the contents of the
            # list of folders
            for contents in content_list[sub_dir]:
                # make the path of the content to move
                path_to_content = sub_dir + "/" + contents
                # make the path with the current folder
                dir_to_move = os.path.join(current_folder, path_to_content)
                # move the file
                shutil.move(dir_to_move, merge_folder_path)
            folder_to_remove = os.path.join(current_folder, sub_dir)
            os.rmdir(folder_to_remove)
    dirpath = os.path.join(raw_data_path, 'omniglot-py/Unified')
    dicts = create_dict(dirpath)
    new_dict = {k: v for k, v in sorted(dicts.items(), key=lambda item: item[1])}

    return new_dict
