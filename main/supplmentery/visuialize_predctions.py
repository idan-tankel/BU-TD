# from supp.training_functions import test_step
# from supp.emnist_dataset import inputs_to_struct
import numpy as np
import matplotlib.pyplot as plt
# from supp.FlagAt import *
# from supp.measurments import get_model_outs
from supplmentery.get_model_outs import get_model_outs
from supplmentery.emnist_dataset import inputs_to_struct
from supplmentery.training_functions import test_step
import torch


def flag_to_comp(flag: torch, ntasks: int) -> tuple:
    """

    :param flag: The flag we wish to compose.
    :param ntasks: The number of tasks we should solve.
    :return: The non zero index entries in the one-hot vectors.
    """
    flag_task = flag[:ntasks]  # The ntasks first entries are related to the task.
    flag_arg = flag[ntasks:]  # The last entries are related to the argument.
    task = torch.argmax(flag_task, dim=0)  # Find the entry in the one hot.
    arg = torch.argmax(flag_arg, dim=0)  # Find the entry in the one hot.
    return task, arg  # return the tuple.


def pause_image(fig=None) -> None:
    """
    Pauses the image until a button is presses.
    """
    plt.draw()
    plt.show(block=False)
    if fig is None:
        fig = plt.gcf()
    fig.waitforbuttonpress()


def visualize(opts, train_dataset):
    """
    visualizes the first batch in the train_dataset.
    :param opts: The model options.
    :param train_dataset: The train_dataset we want to visualize.
    """
    ds_iter = iter(train_dataset)  # iterator over the train_dataset.
    inputs = next(ds_iter)  # The first batch.
    _, outs = test_step(opts, inputs)  # Getting model outs
    model = opts.model
    outs = get_model_outs(model, outs)  # From output to struct.
    samples = inputs_to_struct(inputs)  # From input to struct.
    imgs = samples.image  # Getting the images.
    imgs = imgs.cpu().numpy()  # Moving to the cpu, and transforming to numpy.
    imgs = imgs.transpose(0, 2, 3, 1)  # Transpose to have the appropriate dimensions for an image.
    fig = plt.figure(figsize=(15, 4))  # Defining the plot.
    if opts.use_td_loss:
        n = 3
    else:
        n = 2
    for k in range(len(samples.image)):  # For each image in the batch, show it and its task and label.
        fig.clf()
        fig.tight_layout()
        ax = plt.subplot(1, n, 1)
        ax.axis('off')
        img = imgs[k]  # Getting the k'th image.

        plt.imshow(img.astype(np.uint8))  # Showing the image with the title.
        flag = samples.flag[k]

        adj_type, char = flag_to_comp(flag, opts.ntasks)  # Compose to the task and argument.

        if opts.model_flag is FlagAt.NOFLAG:
            tit = 'Right of all'
        else:
            ins_st = 'The character in the place: %s' % (char.item())
            tit = ins_st
        plt.title(tit)  # Adding the task to the title.

        ax = plt.subplot(1, n, 2)
        ax.axis('off')

        if opts.model_flag is FlagAt.NOFLAG:
            pred = outs.task[k].argmax(axis=0)
            gt = samples.label_task[k]
            gt_st = [cl2let[let] for let in gt]
            pred_st = [cl2let[let] for let in pred]
            font = {'color': 'blue'}
            gt_str = 'Ground Truth:\n%s...' % gt_st[:10]
            pred_str = 'Prediction:\n%s...' % pred_st[:10]
            print(gt_st)
            print(pred_st)
        else:
            gt_val = samples.label_task[k][0].item()  # The ground truth label.
            pred_val = outs.task[k].argmax().item()  # The predicted value.
            if gt_val == pred_val:
                font = {'color': 'blue'}  # If the prediction is correct the color is blue.
            else:
                font = {'color': 'red'}  # If the prediction is incorrect the color is blue.

            gt_str = 'Ground Truth: %s' % gt_val
            pred_str = 'Prediction: %s' % pred_val
            print(char, gt_val, pred_val)
        if opts.use_td_loss:
            tit_str = gt_str
            plt.title(tit_str)
        else:
            tit_str = gt_str + '\n' + pred_str  # plotting the ground truth + the predicted values.
            plt.title(tit_str, fontdict=font)
        plt.imshow(imgs[k].astype(np.uint8))  # Showing the image with the GT + predicted values.
        if opts.use_td_loss:
            ax = plt.subplot(1, n, 3)
            ax.axis('off')
            image_tdk = np.array(outs.td_head[k])
            image_tdk = image_tdk - np.min(image_tdk)
            image_tdk = image_tdk / np.max(image_tdk)
            plt.imshow(image_tdk)
            plt.title(pred_str, fontdict=font)
        print(k)
        label_all = samples.label_all[k]
        print(label_all)
        label_all_chars = [[c for c in row] for row in label_all]
        print(label_all_chars)
        pause_image()
