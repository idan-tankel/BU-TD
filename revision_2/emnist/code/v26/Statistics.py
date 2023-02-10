import csv
import os
import sys

import pandas as pd
import torch
from typing import Tuple

from emnist.code.v26.funcs import get_model_outs, multi_label_accuracy_base


class Constants:
    stats_dir_name = 'stats'


def write_model_output(train_opts, dataset, logger, model, inputs_to_struct, nclasses):
    logger.info("starting write_model_output")
    # Init file
    dir_path = os.path.join(train_opts.model_dir, Constants.stats_dir_name)
    file_name: str = os.path.join(dir_path, get_csv_name_from_int(train_opts.EPOCHS))
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    # Go over inputs
    all_acc = {}
    for inputs in dataset:
        samples = inputs_to_struct(inputs)
        train_opts.model.eval()
        with torch.no_grad():
            outs = train_opts.model(inputs)
            outs = get_model_outs(model, outs)
            preds, task_accuracy = multi_label_accuracy_base(outs, samples)
            for image_id, acc in zip(samples.id, task_accuracy):
                all_acc[str(image_id.item())] = acc.item()

    # Save the output to file
    w = csv.writer(open(file_name, 'w'))
    for image_id, acc in all_acc.items():
        w.writerow([image_id, acc])

    logger.info("finished write_model_output")


def get_csv_name_from_int(epochs):
    return "accuracy_" + str(epochs) + ".csv"


def compare_results(first_path: str, second_path: str, epoch: int):
    first_dict: dict = get_dict_from_file_name(epoch, first_path)
    second_dict: dict = get_dict_from_file_name(epoch, second_path)

    both_true = 0
    both_wrong = 0
    td_true_bu2_false = 0
    td_false_bu2_true = 0

    check_same_keys(first_dict, second_dict)
    count = len(first_dict.keys())
    for image_id in first_dict:
        if first_dict[image_id]:
            if second_dict[image_id]:
                both_true += 1
            else:
                td_true_bu2_false += 1

        else:
            if second_dict[image_id]:
                td_false_bu2_true += 1
            else:
                both_wrong += 1

    # print output
    print((
            'End statistics {}, '
            '\nboth_true: {} {:.2f}%, '
            '\nboth_wrong: {} {:.2f}%, '
            '\ntd_false_bu2_true: {} {:.2f}%, '
            '\ntd_true_bu2_false: {} {:.2f}%').format(
            count,
            both_true, both_true * 100.0 / count,
            both_wrong, both_wrong * 100.0 / count,
            td_false_bu2_true, td_false_bu2_true * 100.0 / count,
            td_true_bu2_false, td_true_bu2_false * 100.0 / count))
    # second_dir_full_path = os.path.join('../../', 'data', 'results', second_path, Constants.stats_dir_name,
    #                                     get_csv_name_from_int(epoch))
    # second_file: pd.DataFrame = pd.read_csv(second_dir_full_path, names=columns_names)


def check_same_keys(first_dict, second_dict):
    # check same keys
    first_keys = list(first_dict.keys())
    second_keys = list(second_dict.keys())
    first_keys.sort()
    second_keys.sort()
    if not first_keys == second_keys:
        print("They don't have the same keys")
        sys.exit()


def get_dict_from_file_name(epoch, first_path):
    columns_names: Tuple[str, str] = ('id', 'right')
    first_dir_full_path = os.path.join('../../', 'data', 'results', first_path, Constants.stats_dir_name,
                                       get_csv_name_from_int(epoch))
    if not os.path.exists(first_dir_full_path):
        print('No such file: ' + first_dir_full_path)
        sys.exit()
    first_file: pd.DataFrame = pd.read_csv(first_dir_full_path, names=columns_names)
    first_dict = dict(zip(first_file.id, first_file.right))
    return first_dict


if __name__ == "__main__":
    compare_results("emnist_pyt_v26_flagat_bu2_sgd0_two_losses_6_sufficient",
                    'emnist_pyt_v26_flagat_td_sgd0_two_losses_6_sufficient',
                    200)
