def main_FashionEmnist(language_idx,train_right,train_left):
    opts = Model_Options_By_Flag_And_DsType(Flag = Flag.SF, DsType = DsType.FashionMnist)
    parser = GetParser(opts = opts, language_idx = language_idx)
    print_detail(parser)
    data_path = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/FashionMnist/samples/6_extended_testing_' + str(language_idx)
    # Create the data for right.
    [the_datasets, _, _,test_dl,_, _, _] = get_dataset_for_spatial_realtions(parser, data_path,embedding_idx =  0, direction =  0)
    if train_right:
        parser.EPOCHS = 20
        training_flag = Training_flag(train_all_model=True, train_arg=True, task_embedding=True, head_learning=True)
        train_omniglot(parser, embedding_idx = 0, the_datasets=the_datasets, training_flag=training_flag)

    if train_left:
        parser.EPOCHS = 100
        [the_datasets, _, _, _,_, _, _] = get_dataset_for_spatial_realtions(parser, data_path, embedding_idx =  1, direction =  1)
        training_flag = Training_flag(train_all_model=False, train_arg=False, task_embedding=True, head_learning=True)
        train_omniglot(parser, embedding_idx = 1, the_datasets=the_datasets, training_flag=training_flag)
    print("Done training left, with accuracy : " + str(acc))

def train_cifar10(parser, embedding_idx, the_datasets):
    set_datasets_measurements(the_datasets, Measurements, parser, parser.model)
    cudnn.benchmark = True  # TODO:understand what it is.
    # Deciding which parameters will be trained: if True all the model otherwise,only the task embedding.
    learned_params = parser.model.parameters()
    # Training the learned params of the model.
    return train_model(parser, the_datasets, learned_params, embedding_idx)

def main_cifar10(lr=0.01, wd=0.0, lr_decay=1.0, language_idx=0):
    parser = GetParser(DsType.Cifar10, lr, wd, lr_decay, language_idx, use_bu1_loss=False)
    print_detail(parser)
    data_path = '/home/sverkip/data/BU-TD/yonathan/training_cifar10/data/processed'
    # Create the data for right.
    [the_datasets, _, test_dl, _, _] = get_dataset_cifar(parser, data_path)
    return train_cifar10(parser, embedding_idx=0, the_datasets=the_datasets)

def main_emnist(language_idx,train_right,train_left):
    opts = Model_Options_By_Flag_And_DsType(Flag = Flag.SF, DsType = DsType.Emnist)
    parser = GetParser(opts = opts, language_idx = language_idx)
    print_detail(parser)

    data_path = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/samples/6_extended' + str(language_idx)
    # Create the data for right.
    [the_datasets, _, _,test_dl,_, _, _] = get_dataset_for_spatial_realtions(parser, data_path,embedding_idx =  0, direction =  0)
    # Training Right.
    if train_right:
        parser.EPOCHS = 20
        training_flag = Training_flag(train_all_model=True, train_arg=True, task_embedding=True, head_learning=True)
        train_omniglot(parser, embedding_idx = 0, the_datasets=the_datasets, training_flag=training_flag)

    if train_left:
        parser.EPOCHS = 100
        [the_datasets, _, _, _,_, _, _] = get_dataset_for_spatial_realtions(parser, data_path, embedding_idx =  1, direction =  1)
        training_flag = Training_flag(train_all_model=False, train_arg=False, task_embedding=True, head_learning=True)
        train_omniglot(parser, embedding_idx = 1, the_datasets=the_datasets, training_flag=training_flag)
    print("Done training left, with accuracy : " + str(acc))