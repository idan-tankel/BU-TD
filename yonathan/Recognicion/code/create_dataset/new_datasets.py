from PIL import Image
import torchvision
download_dir = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/FashionMnist'
train_raw_dataset = torchvision.datasets.SVHN(download_dir, split = 'extra', download=True)
print(len(train_raw_dataset))
print(train_raw_dataset[0])
im = train_raw_dataset[0][0]
#print(im.shape)
im.show()
labels =[]
for i in range(len(train_raw_dataset)):
    labels.append(train_raw_dataset[i][1])
print(len(set(labels)))




