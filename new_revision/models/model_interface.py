# interface to some model classes
def load_data_interface(batch,batch_index,task):
    if task:
        x, y = batch['img'], batch['label_task']
    else:
        x, y = batch['img'], batch['label_all']
def loss_interface():
    pass
