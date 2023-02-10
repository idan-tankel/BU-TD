from datetime import datetime
from typing import List
import logging
from types import SimpleNamespace
import numpy as np

from persons.code.v26.funcs import test_step, from_network, get_predicted_feature_value
from persons.code.v26.models.DatasetInfo import DatasetInfo, create_permutations, \
    change_input_according_current_permutations, change_input_according_current_permutations2, create_permutations2
from persons.code.v26.Configs.ConfigClass import ConfigClass


def from_input_list_to_struct(inputs):
    input_at_struct = SimpleNamespace()
    input_at_struct.images = inputs[0]
    input_at_struct.avatars = inputs[1]
    input_at_struct.flags = inputs[2]
    # input_at_struct.?/ = inputs[3]
    input_at_struct.flag_gt = inputs[4]
    input_at_struct.maybe_id = inputs[5]
    input_at_struct.all_image_gt = inputs[6]
    input_at_struct.flags_in_image = inputs[7]  # including all persons?
    input_at_struct.person_in_flag_all_gt = inputs[8]

    return input_at_struct


class LaunchStatistics:
    def __init__(self, datasets, config, opts, inputs_to_struct, logger):
        self.datasets: List[DatasetInfo] = datasets
        self.config: ConfigClass = config
        self.opts = opts
        self.inputs_to_struct = inputs_to_struct
        self.logger = logger

    def calculate_statistics(self):
        location_flags = 2
        location_GT = 4
        # wanted_dataset = [dataset for dataset in self.datasets if dataset.name == self.config.Statistics.wanted_dataset]
        wanted_dataset = self.datasets

        both_true = 0
        both_wrong = 0
        sep_true_tog_false = 0
        sep_false_tog_true = 0

        # go over the wanted data set(train)
        count = 0
        start = datetime.now()
        for dataset_index, curr_dataset in enumerate(wanted_dataset):
            for input_index, inputs in enumerate(curr_dataset.dataset):
                inputs_struct = from_input_list_to_struct(inputs)

                permutations = create_permutations2(inputs_struct, 6,
                                                    number_of_wanted_tasks=self.config.Statistics.number_of_tasks)
                # in create permutation use: inputs_struct.person_in_flag_all_gt to create the wanted permutations. + add some inputs that are 8.
                n_outs_seperatly = []
                locations_for_permutations = list(range(
                    self.config.Statistics.number_of_tasks))  # This if not fort the use of the all flag - but only the task - noty the person

                # First handle one 'together' - then try the statistics
                for curr_flag_number in locations_for_permutations:
                    change_input_according_current_permutations2(curr_flag_number, inputs, permutations,
                                                                 location_GT, location_flags)
                    # use - test_step - (no learning) - to get the output for 'n' of:
                    # inputting 1 by one options
                    # inputting all n instructions
                    loss, outs = test_step(inputs, self.opts)
                    samples, outs = from_network(inputs, outs, self.opts.model.module, self.inputs_to_struct)
                    batch_seperatly_correct = []
                    for curr_sample_index in list(range(inputs[0].shape[0])):
                        # predicted_feature_value = get_predicted_feature_value(outs.task, inputs[location_flags],
                        #                                                       curr_sample_index)
                        predicted_feature_value = outs.task[curr_sample_index][0]
                        curr_gt = inputs[location_GT][curr_sample_index]
                        batch_seperatly_correct.append(predicted_feature_value.argmax() == curr_gt.item())
                    # curr_outputs = [[(index_layer, index_in_layer, curr.argmax())
                    #                  for index_in_layer, curr in enumerate(curr_layer)]
                    #                 for index_layer, curr_layer in enumerate(outs.task) if len(curr_layer) > 0]
                    n_outs_seperatly.append(batch_seperatly_correct)
                # sort
                # n_outs_seperatly.sort(key=lambda x: x[0])
                # now do it for all together
                change_input_according_current_permutations2(self.config.Statistics.number_of_tasks, inputs,
                                                             permutations,
                                                             location_GT, location_flags)
                # n_outs_together
                _, n_outs_together = test_step(inputs, self.opts)
                _, n_outs_together = from_network(inputs, n_outs_together, self.opts.model.module,
                                                  self.inputs_to_struct)
                n_outs_together_1 = []
                for curr_sample_index in list(range(inputs[0].shape[0])):
                    # predicted_feature_value = get_predicted_feature_value(n_outs_together.task, inputs[location_flags],
                    #                                                       curr_sample_index)
                    predicted_feature_value = [curr.argmax() for curr in n_outs_together.task[curr_sample_index]]
                    curr_gt = inputs[location_GT][curr_sample_index]
                    n_outs_together_1.append(np.array(predicted_feature_value) == np.array(curr_gt.tolist()))

                # n_outs_together = [(index, curr.argmax()) for index, curr in enumerate(n_outs_together.task) if
                #                    len(curr) > 0]

                # sorted_gt = permutations[0][self.config.Statistics.number_of_tasks][1]
                # # check how much is correct and how much is false
                # correct_together = [curr_gt == curr_out[1] for curr_gt, curr_out in zip(sorted_gt, n_outs_together)]
                # correct_seperatly = [curr_gt == curr_out[1] for curr_gt, curr_out in
                #                      zip(sorted_gt, n_outs_seperatly)]
                # TODO - now compare n_outs_together_1:n_outs_seperatly - save them, print them
                for i in list(range(len(n_outs_seperatly))):
                    for j in list(range(len(n_outs_seperatly[0]))):
                        count += 1
                        if n_outs_seperatly[i][j]:
                            if n_outs_together_1[j][i]:
                                both_true += 1
                            else:
                                sep_true_tog_false += 1

                        else:
                            if n_outs_together_1[j][i]:
                                sep_false_tog_true += 1
                            else:
                                both_wrong += 1
                self.logger.info((
                    'dataset: {}  {}#, '
                    'input: {} , '
                    'both_true: {} {:.2f}%, '
                    'both_wrong: {} {:.2f}%, '
                    'sep_false_tog_true: {} {:.2f}%, '
                    'sep_true_tog_false: {} {:.2f}%').format(
                    curr_dataset.name, dataset_index,
                    input_index,
                    both_true, both_true * 100.0 / count,
                    both_wrong, both_wrong * 100.0 / count,
                    sep_false_tog_true, sep_false_tog_true * 100.0 / count,
                    sep_true_tog_false, sep_true_tog_false * 100.0 / count))
        self.logger.info((
            'End statistics {}, '
            'both_true: {} {:.2f}%, '
            'both_wrong: {} {:.2f}%, '
            'sep_false_tog_true: {} {:.2f}%, '
            'sep_true_tog_false: {} {:.2f}%').format(
            count,
            both_true, both_true * 100.0 / count,
            both_wrong, both_wrong * 100.0 / count,
            sep_false_tog_true, sep_false_tog_true * 100.0 / count,
            sep_true_tog_false, sep_true_tog_false * 100.0 / count))
        self.logger.info("took: " + str(datetime.now() - start))
        # save the statistics - in directory - that creates in the launch time with launch time name
        # test_step()
