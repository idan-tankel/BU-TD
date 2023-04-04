# BU-TD
## Official code for the paper [Image interpretation by iterative bottom-up top-down processing](https://arxiv.org/abs/2105.05592)
[![arXiv](https://img.shields.io/badge/arXiv-2108.00946-b31b1b.svg)](https://arxiv.org/abs/2105.05592)

Scene understanding requires the extraction and representation of scene components, such as objects and their parts, people, and places, together with their individual properties, as well as relations and interactions between them. We describe a model in which meaningful scene structures are extracted from the image by an iterative process, combining bottom-up (BU) and top-down (TD) networks, interacting through a symmetric bi-directional communication between them (‘counter-streams’ structure). The BU-TD model extracts and recognizes scene constituents with their selected properties and relations, and uses them to describe and understand the image.
![Counter stream](/figures/Counter-stream.png)

Currently the repository contains the code for the Persons and EMNIST experiments (described in Sections 3 and 5 of the paper).
The code creates the data sets used in the paper and also the bottom up (BU) - top down (TD) network model (counter stream).


## Code
The code is based on Python 3.6 and uses PyTorch (version 1.6) as well as torchvision (0.7). Newer versions would probably work as well.
Requirements are in requirements.txt and can also be installed by:

`conda install matplotlib scikit-image Pillow`

For image augmentation also install:

`conda install imgaug py-opencv`

## Persons details
![persons](/figures/persons.png)

Download the raw Persons data (get it [here](https://www.dropbox.com/s/whea9na512vdjvh/avatars_6_raw.pkl?dl=0) and place it in `persons/data/avatars`).

Next, run the following from within the `persons/code` folder. 
Create the sufficient data set:

`python create_dataset.py`

and the extended data set (use `-e`):

`python create_dataset.py -e`

the data sets will be created in the `data` folder.

Run the training code for the sufficient set (`-e` for the extended set):

`python avatar_details.py [-e]`

A folder with all the learned models and a log file will be created under the `data/results` folder.

## EMNIST spatial relations
![emnist](/figures/emnist.png)

Run from within the `emnist/code` folder. 
Create the sufficient data set (`-e` for the extended set) with either 6 or 24 characters in each image (`-n 6` or `-n 24`):

`python create_dataset.py -n 24 -e`

The EMNIST raw dataset will be downloaded and processed (using torchvision) and the spatial data set will be created in the `data` folder.

Run the training code for the sufficient set (using `-e` for the extended set and the corresponding `-n`):

`python emnist_spatial.py -n 24 -e`

A folder with all the learned models and a log file will be created under the `data/results` folder.

## Extracting scene structures
Code will be added soon.

## Paper
If you find our work useful in your research or publication, please cite our work:

[Image interpretation by iterative bottom-up top-down processing](https://arxiv.org/abs/2105.05592)

Shimon Ullman, Liav Assif, Alona Strugatski, Ben-Zion Vatashsky, Hila Levi, Aviv Netanyahu, Adam Yaari


## New version specifics - locations and instructions

In the `alpha` version, the most important things have being moved to the `main` folder


-  *Configs* contain the running configuration. this is a good place to start explore the model parameters and specifications
- *create* is used to create the dataset and the data loader. Since we have used before a custom dataset of EMNIST + relations, we have used a wrapper for the "right / left / up /down" and creating a good descriptors for this in our data
- *supplmentery* contains the main code. These are things like model and block definitions, losses, norms, visualization utilities,...
- *other files* (like `emnist_spatial.py`) are the "trainers" - contains the actual training and evaluation code


## Pytorch lightning trainer

The usual training process has a trining loop and test loop under `training_functions`.
There is a newer option of pytorch lightning version, although it's had a few things to customize (the wandb root for instance)


### The structure of a single example from the dataset
*In this version, any data point is being coverted from a tuple to SimpleNamespace and the opposite*
```python
image = inputs[0], label_existence = inputs[1], label_all = inputs[2], label_task = inputs[3], id = inputs[4], flag = inputs[5]
```
The attributes of a "datapoint"
- `image`: The image followed by transforms and tensored. This can be the image your currnt dataloader loads.
- `label_existance (batch,number_of_classes)`: one hot encoding of which objects are available in the image. Ex: which characters are in EMNIST image. Usually batched
- `label_all (batch,no_of_objects_height,no_of_objects_width)`: A map of the labeled sequence of the image. Usually has shape (6,1)
6 4 8 4 5 8
1 2 3 17 42 
- `label_task (batch,number_of_tasks)`: One hot encoding of the task for each example (batched). usually there are 4 tasks (right,left,up,down)
- `id`: The example ID
- `flag (batch,number_of_tasks + number_of_classes)`: concatenated version of task label_task and argument. The concatination is by design for the instructions embedding of the task and it's argument. There is 1 for the task and 1 for the relevant labeled argument, and all the rest are zeros (for each row).


### Things to consider / Issues
- You may use your own dataset within the trainer, but it must contain some attributes in this version. There are 2 functions [inputs_to_struct](https://github.com/idan-tankel/BU-TD/blob/50fc829b9128e0a62991c595f3f8c628c2302293/main/supplmentery/emnist_dataset.py#L94) and the opposite `structs_to_input` who would decompose this. The above is true also when using the raw cnn model - since the flags must by design follow some rules. A suggestion
 - Follow the structure inputs constructions of the EMNIST dataset with relations
 - Create a custom loader of your own dataset using the `inputs_to_structs` and the `structs_to_inputs` functions
 - There are a few specifications of `DsType` (supported datasets). You may see this as a depracated. The only thing you need to follow is adding a sturctured dataloader for the forward of the model
 - If there is any other issues, please open them up within the repository :-) 






## TODO 
- change the design of `create_dataset` to be one big file with plugins to each dataset 
    - emnist
    - persons
    - clevr
    - omniglot
- change the `create_dataset` file to be idempotent - namely, when running the file after the dataset is cretaed it won't be downloaded again 
- change the behaviour of the download function path to not be relative

- using the AllOptions under `Dataset_and_model_type_specifications`, create a few "training profiles" for a several supported datasets, with a defined options. As a result, by specifiying only the dataset name, you will get a set of default options initialized


 - PLAN
    - The `beta` branch will be the major branch, chaging the main
    - Create a packages from `supplmentery` dir and `v26` dir
    - End up with a nice robust base version for the model
    - Scan the code for problems
    - Update documentation to the new training procedures
    - create new `requirements` file for conda
    

- Long Time forward
    - [ ] Create a workflow and a proper GitHUB action to invoke training / debugging / simple auto tests
    - [ ] Transfer the use case of some np.array to torch framework
    - [ ] Slurm integration https://pytorch-lightning.readthedocs.io/en/stable/clouds/cluster_advanced.html


## Contribute

@yonatansverdlov @idan-tankel @liavassif
