import torchvision.datasets as datasets

flowers = datasets.Flowers102(root='/home/sverkip/data/raw', download=False)
birds = datasets
print(flowers[0])
print(len(flowers))