import random
import time
from typing import List, Union

import numpy as np
# from line_profiler_pycharm import profile
import torch
import itertools

from common import logger
from common.common_classes import DatasetInfoBase
# from persons.code.v26.funcs import train_step, test_step, logger
from common.common_functions import train_step, test_step
from persons.code.v26.Configs.ConfigClass import ConfigClass
from persons.code.v26.ConstantsBuTd import get_dev


def get_multiple_gt(curr_gt_list: List[int], wanted_features: List[int], index_start_person_in_gt_list: int):
    gt = []
    wanted_features = wanted_features.copy()
    wanted_features.sort()
    [gt.append(curr_gt_list[index_start_person_in_gt_list + curr_feature]) for curr_feature in wanted_features]
    return gt


def create_permutations2(inputs_struct, all_real_values: int, number_of_wanted_tasks: int = -1):
    """
        Creates permutations of the inputs.
        :param inputs:
        :param all_real_values: location of the GT(for all)
        :param number_of_tasks: number of wanted to be combined tasks - if -1 - non combined
        :return: list of relevant tuples of: flags, GT
        """
    number_of_persons, number_of_tasks, permutations, permutations_one = init_permutation_lists()
    # duplicate - number of examples
    # shuffle each separately
    for curr_example_index in list(range(inputs_struct.avatars.shape[0])):
        # permutation
        curr_flag = permutations_one.copy()
        curr_gt_list = inputs_struct.all_image_gt[
            curr_example_index].tolist()  # Check those locations are the same as curr_flag

        if number_of_wanted_tasks > 1:
            # Change the permutations to be: length of number_of_tasks+1, and for all the same person(random
            #   from what they have), and the last one - has all of their flags(features - and too same person)
            # Create all the separate flags
            wanted_features = random.sample(range(0, number_of_tasks), number_of_wanted_tasks)
            all_curr_flags = inputs_struct.person_in_flag_all_gt[curr_example_index]

            wanted_features.sort()
            curr_permutations = []
            curr_flag_combined = inputs_struct.flags[curr_example_index].clone()
            curr_flag_combined[6:] = 0
            curr_gt_combined = []

            for curr_wanted_feature in wanted_features:
                # separate flags
                curr_flag = inputs_struct.flags[curr_example_index].clone()
                curr_flag[6:] = 0
                curr_flag[6 + curr_wanted_feature] = 1

                curr_permutations.append((curr_flag, all_curr_flags[curr_wanted_feature]))
                # combined
                curr_flag_combined[6 + curr_wanted_feature] = 1
                curr_gt_combined.append(all_curr_flags[curr_wanted_feature])
            # combined
            curr_permutations.append((curr_flag_combined, curr_gt_combined))

            # wanted_person = random.randint(0, number_of_persons - 1)
            # [curr_permutations.append((create_one_flag(wanted_person, curr_feature),
            #                            curr_gt_list[wanted_person * number_of_tasks + curr_feature])) for curr_feature
            #  in
            #  wanted_features]
            # Create all the combined flag
            # curr_permutations.append((create_one_flag(wanted_person, wanted_features),
            #                           get_multiple_gt(curr_gt_list, wanted_features, wanted_person * number_of_tasks)))

            permutations.append(curr_permutations)
        else:
            # shuffle
            temp = list(zip(curr_flag, curr_gt_list))
            random.shuffle(temp)

            curr_flag, curr_gt = zip(*temp)
            # append as tuple
            permutations.append((curr_flag, curr_gt))

    return permutations


def create_permutations(inputs, all_real_values: int, number_of_wanted_tasks: int = -1):
    """
        Creates permutations of the inputs.
        :param inputs:
        :param all_real_values: location of the GT(for all)
        :param number_of_tasks: number of wanted to be combined tasks - if -1 - non combined
        :return: list of relevant tuples of: flags, GT
        """
    number_of_persons, number_of_tasks, permutations, permutations_one = init_permutation_lists()
    # duplicate - number of examples
    # shuffle each separately
    for curr_example_index in list(range(inputs[0].shape[0])):
        # permutation
        curr_flag = permutations_one.copy()
        curr_gt_list = inputs[all_real_values][
            curr_example_index].tolist()  # Check those locations are the same as curr_flag

        if number_of_wanted_tasks > 1:
            # Change the permutations to be: length of number_of_tasks+1, and for all the same person(random
            #   from what they have), and the last one - has all of their flags(features - and too same person)
            # Create all the separate flags
            wanted_features = random.sample(range(0, number_of_tasks), number_of_wanted_tasks)
            wanted_features.sort()
            wanted_person = random.randint(0, number_of_persons - 1)
            curr_permutations = []
            [curr_permutations.append((create_one_flag(wanted_person, curr_feature),
                                       curr_gt_list[wanted_person * number_of_tasks + curr_feature])) for curr_feature
             in
             wanted_features]
            # Create all the combined flag
            curr_permutations.append((create_one_flag(wanted_person, wanted_features),
                                      get_multiple_gt(curr_gt_list, wanted_features, wanted_person * number_of_tasks)))

            permutations.append(curr_permutations)
        else:
            # shuffle
            temp = list(zip(curr_flag, curr_gt_list))
            random.shuffle(temp)

            curr_flag, curr_gt = zip(*temp)
            # append as tuple
            permutations.append((curr_flag, curr_gt))

    return permutations


# def create_permutations(inputs, all_real_values: int, number_of_wanted_tasks: int = -1):
#     """
#         Creates permutations of the inputs.
#         :param inputs:
#         :param all_real_values: location of the GT(for all)
#         :param number_of_tasks: number of wanted to be combined tasks - if -1 - non combined
#         :return: list of relevant tuples of: flags, GT
#         """
#     number_of_persons, number_of_tasks, permutations, permutations_one = init_permutation_lists()
#     # duplicate - number of examples
#     # shuffle each separately
#     for curr_example_index in list(range(inputs[0].shape[0])):
#         # permutation
#         curr_flag = permutations_one.copy()
#         curr_gt_list = inputs[all_real_values][
#             curr_example_index].tolist()  # Check those locations are the same as curr_flag
#
#         if number_of_wanted_tasks > 1:
#             # Change the permutations to be: length of number_of_tasks+1, and for all the same person(random
#             #   from what they have), and the last one - has all of their flags(features - and too same person)
#             # Create all the separate flags
#             wanted_features = random.sample(range(0, number_of_tasks), number_of_wanted_tasks)
#             wanted_features.sort()
#             wanted_person = random.randint(0, number_of_persons - 1)
#             curr_permutations = []
#             [curr_permutations.append((create_one_flag(wanted_person, curr_feature),
#                                        curr_gt_list[wanted_person * number_of_tasks + curr_feature])) for curr_feature
#              in
#              wanted_features]
#             # Create all the combined flag
#             curr_permutations.append((create_one_flag(wanted_person, wanted_features),
#                                       get_multiple_gt(curr_gt_list, wanted_features, wanted_person * number_of_tasks)))
#
#             permutations.append(curr_permutations)
#         else:
#             # shuffle
#             temp = list(zip(curr_flag, curr_gt_list))
#             random.shuffle(temp)
#
#             curr_flag, curr_gt = zip(*temp)
#             # append as tuple
#             permutations.append((curr_flag, curr_gt))
#
#     return permutations


def init_permutation_lists():
    permutations = []
    # use? index_task = torch.where((flag[i])[:6] == 1)[0].item() * 7 + torch.where((flag[i])[6:] == 1)[
    number_of_persons = 6
    number_of_tasks = 7
    permutations_persons = create_one_flag_permutations(number_of_persons)
    permutations_tasks = create_one_flag_permutations(number_of_tasks)
    permutations_one = [person + task for person in permutations_persons for task in permutations_tasks]
    return number_of_persons, number_of_tasks, permutations, permutations_one


def create_one_flag(person: int, features: Union[int, List[int]], number_of_persons: int = 6,
                    number_of_features: int = 7):
    person_list = [0.0] * number_of_persons
    features_list = [0.0] * number_of_features
    person_list[person] = 1
    if isinstance(features, list):
        for curr in features:
            features_list[curr] = 1
    else:
        features_list[features] = 1
    return person_list + features_list


def create_one_flag_permutations(number_of_persons):
    permutations_persons = []
    for index in list(range(number_of_persons)):
        curr = [0.0] * number_of_persons
        curr[index] = 1.0
        permutations_persons.append(curr)
    return permutations_persons


def change_input_according_current_permutations(curr_flag_number, inputs, inputs_permutations, location_GT,
                                                location_flags):
    # if isinstance(curr_flag_number, list):
    #     # check more that one input
    #     # np.array(list(inputs_permutations[0][0]))[curr_flag_number]
    #     # curr_flag = [curr[0][curr_flag_number] for curr in inputs_permutations]
    #     pass
    # else:
    # if isinstance(inputs_permutations[0][1], tuple):
    #     curr_flag = [curr[curr_flag_number][0] for curr in inputs_permutations]
    #     curr_gt = [[curr[curr_flag_number][1]] for curr in inputs_permutations]
    # else:
    curr_flag = [curr[0][curr_flag_number] for curr in inputs_permutations]
    curr_gt = [[curr[1][curr_flag_number]] for curr in inputs_permutations]
    inputs[location_GT] = torch.as_tensor(curr_gt, device=get_dev())
    inputs[location_flags] = torch.as_tensor(curr_flag, device=get_dev())


def change_input_according_current_permutations2(curr_flag_number, inputs, inputs_permutations, location_GT,
                                                 location_flags):
    # if isinstance(curr_flag_number, list):
    #     # check more that one input
    #     # np.array(list(inputs_permutations[0][0]))[curr_flag_number]
    #     # curr_flag = [curr[0][curr_flag_number] for curr in inputs_permutations]
    #     pass
    # else:
    # if isinstance(inputs_permutations[0][1], tuple):
    #     curr_flag = [curr[curr_flag_number][0] for curr in inputs_permutations]
    #     curr_gt = [[curr[curr_flag_number][1]] for curr in inputs_permutations]
    # else:
    curr_flag = [curr[curr_flag_number][0] for curr in inputs_permutations]
    curr_gt = [curr[curr_flag_number][1] for curr in inputs_permutations]
    inputs[location_GT] = torch.as_tensor(curr_gt, device=get_dev())
    if isinstance(curr_flag[0], torch.Tensor):
        inputs[location_flags] = torch.stack(curr_flag)
    else:
        inputs[location_flags] = torch.as_tensor(curr_flag, device=get_dev())


class DatasetInfo(DatasetInfoBase):
    """encapsulates a (train/test/validation) dataset with its appropriate train or test function and Measurement
    class """

    def __init__(self, is_train, ds, nbatches, name, checkpoints_per_epoch=1):
        super().__init__(is_train, ds, nbatches, name, checkpoints_per_epoch)
        if is_train:
            self.batch_fun = train_step
        else:
            self.batch_fun = test_step

    # @profile
    def do_epoch(self, epoch, opts, number_of_epochs, config: ConfigClass):
        aborted, cur_batches, nbatches_report, start_time = super().do_epoch(epoch, opts, number_of_epochs, config)
        location_flags = 2
        location_GT = 4
        for inputs in self.dataset_iter:
            if config.RunningSpecs.all_possible_flags_permutations:
                # Create permutations of the flags for each sample(between different sample shouldn't be the same).
                # for the location 3 and the corresponding GT for location 5
                inputs_permutations = create_permutations(inputs, all_real_values=6)

                for curr_flag_number in list(range(len(inputs_permutations[0][0]))):
                    change_input_according_current_permutations(curr_flag_number, inputs, inputs_permutations,
                                                                location_GT, location_flags)

                    self.run_single_batch(inputs, opts)
            else:
                self.run_single_batch(inputs, opts)
            cur_batches += 1
            template = 'Epoch {}/{} step {}/{} {} ({:.1f} estimated minutes/epoch)'
            if cur_batches % nbatches_report == 0:
                start_time = self.log_end_batch(cur_batches, epoch, nbatches_report, number_of_epochs, start_time,
                                                template)
                if self.is_train and cur_batches > self.number_of_batches:
                    aborted = True
                    break

        if not aborted:
            self.needinit = True
        self.measurements.add_history(epoch)

    def log_end_batch(self, cur_batches, epoch, nbatches_report, number_of_epochs, start_time, template):
        duration = time.time() - start_time
        start_time = time.time()
        estimated_epoch_minutes = duration / 60 * self.number_of_batches / nbatches_report
        logger.info(
            template.format(epoch + 1, number_of_epochs, cur_batches, self.number_of_batches,
                            self.measurements.print_batch(),
                            estimated_epoch_minutes))
        return start_time

    def run_single_batch(self, inputs, opts):
        cur_loss, outs = self.batch_fun(inputs, opts)
        with torch.no_grad():
            # so that accuracies' calculation will not accumulate gradients
            self.measurements.update(inputs, outs, cur_loss.item())
