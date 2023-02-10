import os.path
import torch
import os


def first_free_name() -> str:
    base_name = os.path.join("..", 'data', 'results', 'for_vae')
    if not os.path.exists(base_name):
        os.makedirs(base_name)
    list_dir = os.listdir(base_name)
    if len(list_dir) == 0:
        max_index = 0
    else:
        files_numbers = [os.path.splitext(filename)[0] for filename in list_dir]
        max_index = max([int(x) for x in files_numbers if x.isnumeric()])
    return os.path.join(base_name, str(max_index + 1) + '.pt')


class StoreForVae:
    # the goal is that info will be array where first of last dim is for the number of Samples
    info = []
    it_to_save = False

    def __init__(self, it_to_save=False):
        self.is_to_save = it_to_save

    def add_value(self, value):
        if self.is_to_save:
            self.info.append(value)

    def save_values(self):
        if self.is_to_save:
            torch.save(self.info, first_free_name())
            self.info = []
