import os.path
import pickle
import torchvision

class Metadata:
    def __init__(self,nsamples_train,nsamples_test):
        self.nsamples_train = nsamples_train
        self.nsmaples_test = nsamples_test
    
      

def get_dataset_type(ds_type):
    if ds_type == 0:
     return 'train'
    else:
     return 'test'

def create_cifar10_dataset(download_dir,need_create = True):
    raw_data_download_dir = os.path.join(download_dir,'RAW')
    train_raw_data = torchvision.datasets.CIFAR10(raw_data_download_dir, train=True, download=True)
    test_raw_data = torchvision.datasets.CIFAR10(raw_data_download_dir, train=False, download=True)
    datasets = [train_raw_data, test_raw_data]
    if not os.path.exists(os.path.join(download_dir,'processed/')):
     os.makedirs(os.path.join(download_dir,'processed/'))
     if need_create:
        for ds_type in range(2):
         dataset = datasets[ds_type]
         train = get_dataset_type(ds_type)
         if not os.path.exists(os.path.join(download_dir,'processed/'+train)):
          os.makedirs(os.path.join(download_dir,'processed/'+train))
         for i in range(len(dataset)):
          image = dataset[i][0]
          save_path = os.path.join(download_dir,'processed/'+train+"/"+str(i))
          image.save(save_path+"_img.jpg")
          label = dataset[i][1]
          pickle.dump(label, open(save_path+".pkl", 'wb'))
          print("Done "+str(i)+" image")
    conf_data_fname =os.path.join(os.path.join(download_dir,'processed/'), "Metadata")
    MetaData = Metadata(len(datasets[0]),len(datasets[1]))
    with open(conf_data_fname, "wb") as new_data_file:
        pickle.dump((MetaData), new_data_file)
 
download_dir = '/home/projects/shimon/sverkip/cifar10/data'
create_cifar10_dataset(download_dir)
