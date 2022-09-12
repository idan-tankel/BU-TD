from supplmentery.create_model import create_model
from Configs.Config import Config
from supplmentery.get_dataset import get_dataset
from supplmentery.FlagAt import FlagAt
from supplmentery.Parser import *
from supplmentery import measurments, training_functions, logger, visuialize_predctions
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch import device
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# import supplmentery
# from {Package.module} import {class}


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
dev = device("cuda") if torch.cuda.is_available() else device("cpu")
# this is not used here


def train_emnist(embedding_idx=1, flag_at=FlagAt.SF,
                 processed_data='5_extended', path_loading=None,  train_all_model=True):
    """
    train_emnist This is the main training function under the main function
    This function is used to train the model on the EMNIST dataset
    Stages:
    1. Get the config using the Config class
    2. call dataset using the get_dataset function
    3. create the model using the create_model function (and the flagAt specified)
    4. initialize the learned parameters
    5. train the model using the `training_functions.train function` - that function logs results and all stuff needed
    6. save the model using the training_functions.save_model function

    Args:
        embedding_idx (int, optional): An enumeration to the direction head to take and to train on. Defaults to 0.
        flag_at (`FlagAt`, optional): . This flag determines the architecture specification to use.Defaults to FlagAt.SF.
        processed_data (str, optional): _description_. Defaults to '5_extended'.
        path_loading (`str`, optional): A path to load existing model from with trained weights. Defaults to None.
        train_all_model (bool, optional): Determines which training parameters to initialize. If set to false, only the task_embedding will be initialize. Defaults to True.
    """
    # add some training options from config file
    config: Config = Config()
    config.Models.init_model_options()
    config.flag_size = config.Models.nclasses[0][0] + 3 # directions

    if config.Visibility.interactive_session:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # Getting the options for creating the model and the hyper-parameters.
    results_dir = '../data/emnist/data/results'
    parser = GetParser(flag_at, processed_data, embedding_idx, results_dir)
    parser = config
    # Getting the dataset for the training.
    # TODO initialize all model options from args (see `v26.functions.inits.py` under itsik branch)

    data_path = os.path.join(
        '../data/new_samples', processed_data)
    [the_datasets, train_dl, test_dl, val_dl, train_dataset,
        test_dataset] = get_dataset(direction=embedding_idx, args=parser, data_fname=data_path)
    # Printing the model and the hyper-parameters.
    logger.print_detail(parser)
    # creating the model according the parser.
    model = create_model(model_opts=parser)
    measurments.set_datasets_measurements(datasets=the_datasets, measurements_class=measurments.Measurements, model_opts=parser, model=model)
    cudnn.benchmark = True  # TODO:understand what it is.
    # Loading a pretrained model if exists.
    if path_loading is not None or config.Training.load_existing_path:
        if path_loading is None:
            path_loading = config.Training.path_loading
        training_functions.load_model(parser, results_dir, path_loading,model=model)
    # Deciding which parameters will be trained: if True all the model otherwise,only the task embedding.
    if train_all_model:
        learned_params = model.parameters()
    # Training the learned params of the model.
    # print(accuracy(parser, val_dl))
    training_functions.train_model(
        parser, the_datasets, learned_params, embedding_idx, model)
    visuialize_predctions.visualize(parser, train_dataset)


def main():
    train_emnist(embedding_idx=1, flag_at=FlagAt.TD,
                 processed_data='6_extended', path_loading=None)


if __name__ == "__main__":
    main()

