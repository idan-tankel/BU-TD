"""
Full EMNIST training.
"""
import copy

from training.Data.Parser import GetParser
from training.Data.Structs import Training_flag
from training.Modules.Create_Models import create_model
from training.Modules.Models import BUTDModel, ResNet
from training.Data.Data_params import Flag
from training.training_funcs.training_step import train_step, Show_zero_forgetting
from Data_Creation.src.Create_dataset_classes import DsType
from training.Utils import num_params


def train_trajectory(Train_initial=True, Train_New=False):
    """
    Here we just call the trainer with the specific freezing.
    Returns:

    """
    Models, Data = [], []
    ds_type = DsType.Emnist
    # The flag is continual learning, and BU-TD model with task embedding.
    opts = GetParser(model_flag=Flag.CL, ds_type=ds_type,model_type=BUTDModel)
    model = create_model(opts)
    first_task = opts.data_obj.initial_directions
    epoch = opts.data_obj.epoch_dictionary[first_task]
    training_flag = Training_flag(opts, train_all_model=True)
    if Train_initial:
        new_model, new_data = train_step(opts=opts, model=model, task=first_task, training_flag=training_flag,
                                         ds_type=ds_type, epochs=epoch)
        Models.append(new_model)
        Data.append(new_data)
    #  new_tasks = opts.new_tasks
    new_tasks = [(0, (0,-2))]
    if Train_New:
        for task in new_tasks:
            # Train head, and task embedding.
            training_flag = Training_flag(opts, train_head=True, train_task_embedding=True)
            epoch = 70 # opts.data_obj.epoch_dictionary[task]
            # The new model, data.
            model, data = train_step(opts=opts, model=model, task=task, training_flag=training_flag, ds_type=ds_type,
                                     epochs=epoch)
            # Copy model.
            Models.append(copy.deepcopy(model))
            # Copy data.
            Data.append(data)
    # Showing old tasks performance is not damaged.
    accuracies = Show_zero_forgetting(opts, Models, Data)
    for idx, acc in enumerate(accuracies):
        print(f"The accuracy of the {idx} model is: {str(acc)} \n")
    print("As we can see we have zero forgetting!")


if __name__ == "__main__":
    train_trajectory(Train_initial=False, Train_New=True)
