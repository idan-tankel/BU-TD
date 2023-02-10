from abc import ABC
from typing import Callable


class TrainingOptionsConfig:
    optimizer_name: str = None
    initial_lr: float = None
    momentum: float = None
    weight_decay: float = None
    scale_batch_size: int = None
    num_epochs: int = None
    num_epochs_pretrain: int = None
    checkpoints_per_epoch: int = None
    distributed: bool = None


class ModelSpecifications:
    dim: int = None
    depth: int = None
    heads: int = None
    mlp_dim: int = None
    dropout: float = None
    emb_dropout: float = None


class Model:
    transformer_model_name: str = None  # Short name like: pit
    transformer_model_implementation: str = None  # long name like: pit_ti_224
    transformer_model_input_shape: tuple = None
    number_of_heads: int = None
    dataset_to_use: str = None
    train: bool = None
    inference: bool = None
    is_parallel: bool = None  # If true - should use different folders - should save it with time executed
    training_options_config: TrainingOptionsConfig = TrainingOptionsConfig()
    model_specifications: ModelSpecifications = ModelSpecifications()


class Running:  # TODO should rearrange Running and TrainingOptionsConfig
    batch_size: int = None  # TODO Should move to TrainingOptionsConfig
    num_workers: int = None
    ndirections: int = None
    nsamples_train: int = None
    nsamples_test: int = None
    nsamples_val: int = None
    inshape: (int, int, int) = None
    is_load_model: bool = None
    number_batches_report: int = None
    save_model_every_n_epochs: int = None
    location_to_save_model: str = None
    nsamples_train_quick: int = None
    nsamples_test_quick: int = None
    nsamples_val_quick: int = None
    is_testing_code: bool = None


class RunningSpecification:
    model: Model = Model()
    running: Running = Running()
    is_to_fit: bool = None
    interactive_session: bool = None
    task_name: str = None


class Dataset:
    base_samples_dir: str = None
    nclasses_existence: int = None
    dataset_name: str = None
    chosen_dataset_full_path: str = None
    dataset_function_name: str = None
    dataset_function: Callable = None
    inshape: (int, int, int) = None
    measurements_function: Callable = None
    mean: (float, float, float) = None
    std: (float, float, float) = None


class Emnist(Dataset):
    dataset_name = 'emnist'


class Persons(Dataset):
    dataset_name = 'persons'


class DatasetsSpecs:
    emnist: Emnist = Emnist()
    persons: Persons = Persons()
    chosen_dataset: Dataset = None


class SavedModelSpecifications:
    best_model_saved_name: str = None
    checkpoint: str = None
    current_model_saved_name: str = None
    file_extension: str = None
    checkpoint_extension: str = None
    location_to_save_model: str = None
    is_parallel: bool = None
    is_force_create_Model: bool = None
    directory_to_save_output: str = None
    saved_weights_name: str = None


class GrowingDataSets:
    is_growing_data_sets: bool = None
    location_to_save_output: str = None
    number_of_hops: int = None
    start_datasets_size: int = None
    name_save_models: str = None


class Config:
    running_specification: RunningSpecification = RunningSpecification()
    datasets_specs: DatasetsSpecs = DatasetsSpecs()
    saved_model_specifications: SavedModelSpecifications = SavedModelSpecifications()
    growing_data_sets: GrowingDataSets = GrowingDataSets()
    project_name: str = None
