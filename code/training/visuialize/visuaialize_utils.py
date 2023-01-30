"""
Visualizing predictions utils.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils

from training.Data.Datasets import DataSetBase
from training.Data.Structs import inputs_to_struct


def pause_image() -> None:
    """
    Pauses the image until a button is presses.
    """
    plt.draw()
    plt.show(block=False)
    fig = plt.gcf()
    fig.waitforbuttonpress()


def title(direction: tuple) -> str:
    """
    Args:
        direction: The direction

    Returns: The title.

    """
    direction_str = ''
    dir_x, dir_y = direction
    if dir_x > 0:
        direction_str += 'right-' * dir_x
    if dir_x < 0:
        direction_str += 'left-' * -dir_x
    if dir_y > 0:
        direction_str += 'down-' * dir_x
    if dir_y < 0:
        direction_str += 'up-' * -dir_x
    direction_str = direction_str[:-1]
    return direction_str


def From_id_to_class_Fashion_MNIST():
    """
    Returns the dictionary from class id to class.
    """
    cl2let = dict()
    cl2let[0] = 'T-shirt'
    cl2let[1] = 'Trouser'
    cl2let[2] = 'Pullover'
    cl2let[3] = 'Dress'
    cl2let[4] = 'Coat'
    cl2let[5] = 'Sandal'
    cl2let[6] = 'Shirt'
    cl2let[7] = 'Sneaker'
    cl2let[8] = 'Bag'
    cl2let[9] = 'Ankle boot'
    cl2let[10] = 'Border'
    return cl2let


def From_id_to_class_EMNIST(mapping_fname: str) -> dict:
    """
    From character id to its name.
    Args:
        mapping_fname: The mapping name.

    Returns: The dictionary

    """
    # the mapping file is provided with the EMNIST dataset
    with open(mapping_fname, "r") as text_file:
        lines = text_file.read().split('\n')
        cl2let = [line.split() for line in lines]
        cl2let = cl2let[:-1]
        cl2let = {int(mapi[0]): chr(int(mapi[1])) for mapi in cl2let}
        cl2let[47] = 'Border'
        cl2let[48] = 'Not in image'
        return cl2let


def Add_keypoint(data_set: DataSetBase, sample: inputs_to_struct, image, k):
    """
    Ass the bounding box.
    Args:
        data_set: The test data-set.
        sample: The sample.
        image: The image.
        k: The image id.

    Returns:

    """
    Sample = data_set.get_raw_sample(index=sample.index[k])
    chars = Sample.chars
    q = Sample.query_part_id
    char = chars[q]
    stx = char.stx
    end_x = char.end_x
    sty = char.sty
    end_y = char.end_y
    box = torch.tensor([80, 10, 80, 10], dtype=torch.int).unsqueeze(0)
    key_point_location = torch.tensor([(stx + end_x) // 2, (sty + end_y) // 2]).reshape(1, 1, 2)
    image = torch.tensor(image)
    image = image.reshape((3, 130, 200))
    image = torchvision.utils.draw_bounding_boxes(image=image, boxes=box, fill=True, colors=(255, 255, 0), width=50)
    black = np.zeros((130, 200, 3), dtype=np.uint8)
    #  print(image.shape)
    #  return cv2.circle(img=image/255, center=key_point_location, radius=2, color=(0, 0, 255))
    image = image.reshape(130, 200, 3)
    return image.numpy()
