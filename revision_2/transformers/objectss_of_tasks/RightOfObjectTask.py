import torch
import torchvision
from torchvision.transforms import Resize

from common.common_classes import DataInputs, MeasurementsBase
from emnist.code.v26.emnist_dataset import EMNISTAdjDatasetNew
from transformers.Utils.RightOfUtils import multi_label_accuracy, get_labels_task_right_of
from transformers.models.Measurments import Measurements
from transformers.objectss_of_tasks.ObjectTaskInterface import ObjectTaskInterface


class RightOfObjectTask(ObjectTaskInterface):
    task_dataset: str = 'emnist'
    task_name: str = 'right_of'
    measurements = Measurements
    dataset_function = EMNISTAdjDatasetNew

    def __init__(self, weights_losses=None, number_classes: int = 47):
        super().__init__(weights_losses)
        self.losses = None
        self.weights = None
        self.output_size = number_classes * (number_classes + 1)

    def add_name(self, measurement: MeasurementsBase):
        name = self.task_name + " Accuracy"
        measurement.add_name(name)
        measurement.init_results()

    def reshape_input(self, model_input_shape) -> Resize:
        return torchvision.transforms.Resize(model_input_shape)

    def forward(self, inputs: DataInputs, resize_layer: torch.nn.Module):
        x = self.transforms_resize(inputs.images)
        # TODO - concatenate the instructions

        x = self.made_model_transformer(x)  # The Vit model
        return x

    def loss(self, inputs: DataInputs, outs, number_classes, is_pretrain: bool = None):
        """
        losses function - calculate loss - for each image - then mean over images


        :param outs: output of the model - shape (batch_size, num_classes)
        :param inputs:  inside it the label of the task - shape (batch_size, num_classes)
        :param number_classes: number of classes
        """
        labels = inputs.label_all.squeeze(1)
        losses_task = torch.zeros((labels.shape[0], number_classes))

        # Calculate right of
        labels_task = get_labels_task_right_of(labels, number_classes)

        for k in range(number_classes):
            start_task_index = int(k * (number_classes + 1))
            end_task_index = int(start_task_index + (number_classes + 1))
            taskk_out = outs[:, start_task_index:end_task_index]
            label_taskk = labels_task[:, k]

            # Take only in the right_of
            loss_taskk = self.loss_function(taskk_out, label_taskk)
            loss_taskk = loss_taskk * (
                ((label_taskk != number_classes) +
                 ((label_taskk == number_classes) / 800)).to(torch.float32))

            losses_task[:, k] = loss_taskk

        return losses_task.mean()

    def accuracy(self, outs, inputs: DataInputs, nclasses,
                 is_pretrain: bool = None):  # This for it how to return more than one acc
        _, task_accuracy, only_left_accuracy = multi_label_accuracy(outs, inputs, nclasses)
        return task_accuracy, only_left_accuracy
