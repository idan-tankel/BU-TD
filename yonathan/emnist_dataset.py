import torch.utils.data as data
import torch
import os
import numpy as np
import pickle
import torchvision

# %% EMNIST dataset
''''
   # Downaload and preprocess the emnist.
   # obtain the EMNIST dataset
   emnist_preprocess = SimpleNamespace()
   emnist_preprocess.convertor = 'pytorch'
   if emnist_preprocess.convertor == 'pytorch':
       download_dir = os.path.join(emnist_dir, 'emnist_raw')
       raw_data_fname = os.path.join(emnist_dir, 'emnist-pyt.pkl')
   else:
       download_dir = '../data/emnist/gzip'
       raw_data_fname = os.path.join(emnist_dir, 'emnist-tf.pkl')
   emnist_preprocess.data_fname = raw_data_fname
   emnist_preprocess.download_dir = download_dir
   emnist_preprocess.mapping_fname = os.path.join(emnist_dir, "emnist-balanced-mapping.txt")
   _, labels, total_bins, LETTER_SIZE, IMAGE_SIZE = load_raw_emnist_data(
       emnist_preprocess)
   '''


class EmnistLoader(data.Dataset):
    def __init__(self, data_dir):
        download_raw_data_dir = os.path.join(data_dir, 'emnist_raw')
        raw_data_fname = os.path.join(data_dir, 'emnist-pyt.pkl')
        convert_raw_data_pytorch(download_raw_data_dir, raw_data_fname)
        print('Converting EMNIST raw data using %s' % 'pytorch')
        print('Done converting EMNIST raw data')
        images, labels, IMAGE_SIZE = load_raw_emnist_data(raw_data_fname)

        self.num_labels = len(set(labels))
        self.images = [[] for _ in range(self.num_labels)]
        self.labels = []
        for idx in range(len(labels)):
            index = labels[idx]
            img = images[idx]
            self.images[index].append(img)
            self.labels.append(index)

        self.images = sum(self.images, [])
        self.nclasses = len(set(labels))
        self.num_examples_per_character = len(labels) // self.nclasses

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        image = image
        return image, label


def convert_raw_data_pytorch(download_dir, data_fname):
    emnist_download_dir = download_dir
    emnist_split_type = 'balanced'
    labels = [[] for _ in range(2)]
    imgs = [[] for _ in range(2)]
    for k, train in enumerate((True, False)):
        emnist_data = torchvision.datasets.EMNIST(
            emnist_download_dir,   emnist_split_type,  train=train,   download=True)
        for img, lbl in emnist_data:
            labels[k].append(lbl)
            img = np.array(img)
            img = img.reshape(-1)
            imgs[k].append(img)
    images_train, images_test = map(np.array, imgs)
    labels_train, labels_test = map(np.array, labels)

    with open(data_fname, "wb") as new_data_file:
        pickle.dump((images_train, images_test, labels_train,
                    labels_test),  new_data_file)


# load the raw EMNIST dataset. If it doesn't exist download and process it
def load_raw_emnist_data(data_fname):
    with open(data_fname, "rb") as new_data_file:
        images_train, images_test, labels_train, labels_test = pickle.load(
            new_data_file)
    xs = np.concatenate((images_train, images_test))
    ys = np.concatenate((labels_train, labels_test))

    LETTER_SIZE = 28
    images = xs.reshape(len(xs), LETTER_SIZE, LETTER_SIZE)
    images = images.transpose((0, 2, 1))
    labels = ys
    total_bins = 8
    IMAGE_SIZE = [LETTER_SIZE * 4, LETTER_SIZE * total_bins]

    return images, labels, IMAGE_SIZE

    # if True, generate multiple examples (image,label) pairs from each image, else generate a single example
    # generate multiple examples (image,label) pairs from each image
    # We don't exclude (char,border) or (border,char) pairs as this drastically limits the available valid training examples
    # Therefore, do not query near borders
   # valid_queries = np.arange(sample_nchars)
  #  near_border = np.arange(0, sample_nchars, obj_per_row)
   # valid_queries_right = np.setdiff1d(valid_queries, near_border)
   # valid_queries_left = np.setdiff1d(valid_queries, near_border)
