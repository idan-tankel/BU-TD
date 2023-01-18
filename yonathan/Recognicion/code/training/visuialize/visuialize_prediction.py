"""
Visualizing the predictions.
"""
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from Data_Creation.Create_dataset_classes import DsType  # Import the Data_Creation set types.
from training.Data.Data_params import Flag
from training.Data.Get_dataset import get_dataset_for_spatial_relations
from training.Data.Parser import GetParser
from training.Data.Structs import Task_to_struct, inputs_to_struct
from training.Modules.Create_Models import create_model
from training.Utils import preprocess, Compose_Flag
from training.visuialize.visuaialize_utils import From_id_to_class_Fashion_MNIST, From_id_to_class_EMNIST, title, \
    pause_image


class Visualize:
    """
    Visualizer class
    """

    def __init__(self, opts: argparse, model_test: nn.Module):
        self.opts = opts
        self.model = model_test
        # opts.Data_specific_path
        path = os.path.join(opts.project_path, 'data/Emnist/RAW/emnist-balanced-mapping.txt')
        self.ds_type = opts.ds_type
        self.cl2let = From_id_to_class_EMNIST(
            mapping_fname=path) if self.ds_type is DsType.Emnist else From_id_to_class_Fashion_MNIST()  # The
        # dictionary.

    def __call__(self, direction: tuple, batch_id: int = 0) -> None:
        """
        Visualize the task.
        Args:
            direction: The direction
            batch_id: The batch id

        """
        project_dir_path = Path(__file__).parents[3]
        data_path = os.path.join(project_dir_path, f'data/{str(self.ds_type)}/samples/(4,4)_Image_Matrix')
        DataLoaders = get_dataset_for_spatial_relations(parser, data_path,
                                                        list_task_structs=[Task_to_struct(0, direction)])
        ds_iter = iter(DataLoaders['test_dl'])  # iterator over the train_dataset.
        inputs = next(ds_iter)
        for _ in range(batch_id):
            inputs = next(ds_iter)  # The first batch.
        model.eval()  # Move to eval mode.
        inputs = preprocess(inputs, self.opts.device)  # Move to device.
        samples = inputs_to_struct(inputs)  # From input to struct.
        outs = model.forward_and_out_to_struct(samples)  # Getting model_test outs and make struct.
        images = samples.image  # Getting the images.
        images = images.cpu().numpy().astype(np.uint8)  # Moving to the cpu, and transforming to numpy.
        images = images.transpose(0, 2, 3, 1)  # Transpose to have the appropriate dimensions for an image.
        fig = plt.figure(figsize=(7, 7))  # Defining the plot.
        flags = samples.flags  # The flags.
        # Compose to the argument flag.
        _, _, arg_flag = Compose_Flag(opts=parser, flags=flags)
        gt_vals = samples.label_task  # The GT.
        pred_vals = outs.classifier.argmax(dim=1)  # The predicted value.
        direction_str = title(direction)  # The title.
        chars = arg_flag.argmax(dim=1)  # All characters.
        for k in range(len(samples.image)):  # For each image in the batch, show it and its list_task_structs and label.
            fig.clf()
            fig.tight_layout()
            ax = plt.subplot(1, 2, 1)
            ax.axis('off')
            img = images[k].astype(np.uint8)  # Getting the k image.
            # img = self.Add_keypoint(data_set=DataLoaders['test_ds'], sample=samples, image=img, k=k)
            plt.imshow(img.astype(np.uint8))  # Showing the image with the title.
            char = self.cl2let[chars[k].item()]  # Current character.
            if self.opts.model_flag is Flag.NOFLAG:
                tit = 'Right of all'
            else:
                tit = 'The character in the place: %s' % char + '\n what is the character in the {}?'.format(
                    direction_str)
            plt.title(tit)  # The title.
            ax = plt.subplot(1, 2, 2)
            ax.axis('off')
            if self.opts.model_flag is Flag.NOFLAG:
                pred = outs.classifier[k].argmax(axis=0)
                gt = samples.label_task[k]
                gt_st = [self.cl2let[let.item()] for let in gt]
                pred_st = [self.cl2let[let.item()] for let in pred]
                font = {'color': 'blue'}
                gt_str = 'Ground Truth:\n%s...' % gt_st[:10]
                pred_str = 'Prediction:\n%s...' % pred_st[:10]
            else:
                gt_val = self.cl2let[gt_vals[k].item()]  # Current sample.
                pred_val = self.cl2let[pred_vals[k].item()]  # Current prediction.
                if gt_val == pred_val:
                    font = {'color': 'blue'}  # If the prediction is correct the color is blue.
                else:
                    font = {'color': 'red'}  # If the prediction is incorrect the color is red.

                gt_str = f'Ground Truth: {gt_val}'
                pred_str = f'Prediction: {pred_val}'

            tit_str = gt_str + '\n' + pred_str  # plotting the ground truth + the predicted values.
            plt.title(tit_str, fontdict=font)
            plt.imshow(images[k].astype(np.uint8))  # Showing the image with the GT + predicted values.
            pause_image()


parser = GetParser(model_flag=Flag.CL)
model = create_model(parser)
# load_model(model_test = model_test, results_dir = parser.results_dir, model_path =
# 'left_success/BUTDModel_best_direction=[( -1, 0)].pt')
project_path = Path(__file__).parents[2]
vis = Visualize(opts=parser, model_test=model)
vis(direction=(-2, 0), batch_id=2)
