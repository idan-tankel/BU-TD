import torchvision.datasets as datasets

flowers = datasets.Food101(root='/home/sverkip/data/raw', download=True)

print(len(flowers))