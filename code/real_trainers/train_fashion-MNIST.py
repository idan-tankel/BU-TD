"""
Full Fashion-MNIST training.
"""
import copy

from training.Data.Parser import GetParser
from training.Data.Structs import Training_flag
from training.Modules.Create_Models import create_model
from training.Modules.Models import *
from training.training_funcs.training_step import train_step, Show_zero_forgetting


def train_trajectory():
    """
    Here we just call the trainer with the specific freezing.
    Returns:

    """
    Models, Data = [], []
    ds_type = DsType.Fashionmnist
    # The flag is continual learning, and BU-TD model with task embedding.
    opts = GetParser(model_flag=Flag.CL, ds_type=ds_type)
    model = create_model(opts)
    first_task = opts.initial_directions
    epoch = opts.epoch_dictionary[first_task[0].unified_task]
    training_flag = Training_flag(opts, train_all_model=True)
    new_model, new_data = train_step(opts=opts, model=model, task=first_task, training_flag=training_flag,
                                     ds_type=ds_type, epochs=epoch)
    Models.append(new_model)
    Data.append(new_data)
    new_tasks = opts.new_tasks
    for task in new_tasks:
        # Train head, and task embedding.
        training_flag = Training_flag(opts, train_head=True, train_task_embedding=True)
        epoch = opts.epoch_dictionary[task[0].unified_task]
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
    train_trajectory()
