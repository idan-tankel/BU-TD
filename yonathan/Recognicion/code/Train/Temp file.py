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




'''
path_loading = 'Model_without_bias/model_latest_left.pt'
model_path = parser.results_dir
load_model(parser, model_path, path_loading, load_optimizer_and_schedular=False);
load_running_stats(parser.model, task_emb_id = 1);
acc = accuracy(parser, test_dl)
print("Done training right, with accuracy : " + str(acc))
'''


def main_emnist(language_idx,direction_idx,train_right,train_left,wd):
    opts = Model_Options_By_Flag_And_DsType(Flag = Flag.SF, DsType = DsType.Emnist)
    path = "Model_right_wd="+str(wd)
    parser = GetParser(opts = opts, language_idx = language_idx,wd=wd,model_path=path)
    print_detail(parser)

    data_path = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/emnist/samples/6'
    # Create the data for right.
    [the_datasets, _, _,test_dl,_, _, _] = get_dataset_for_spatial_realtions(parser, data_path, lan_idx=0, direction_idx=0)
    # Training Right.

    path_loading = 'Model_right_wd=e-4/model_best_right.pt'
    model_path = parser.results_dir
    load_model(parser.model_old, model_path, path_loading, load_optimizer_and_schedular=False);
  #  acc = accuracy(parser.model_old, test_dl)
 #   print("Done training right, with accuracy : " + str(acc))
    if train_right:
        parser.EPOCHS = 60
        training_flag = Training_flag(train_all_model = True, train_arg=True,direction_emb = False,lang_emb = True, head_learning=True)
        train_omniglot(parser, lang_id = 0,direction_id=direction_idx, the_datasets=the_datasets, training_flag=training_flag)
    reg = Regulizer(40,parser.model, test_dl)
    print(reg.fisher)
    if train_left:
        parser.EPOCHS = 100
        [the_datasets, _, _, test_dl, _, _, _] = get_dataset_for_spatial_realtions(parser, data_path, lan_idx=1, direction_idx=1)
        training_flag = Training_flag(train_all_model=False, train_arg=False, direction_emb=False, lang_emb=True, head_learning=True)
        train_omniglot(parser, lang_id=1, direction_id = 1, the_datasets=the_datasets, training_flag=training_flag)
    print("Done training left, with accuracy : " + str(acc))