#read first frame of every data entry.
import torch.utils.data.dataloader as DataLoader
import os as OS
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataset import ConcatDataset

#instructions:
#1. generate keyvalue pairs (assign an index to each first frame of data)
#---
class OurDataset(Dataset):
    datasetPath = '../20bn-jester-v1'
    transform = transforms.Compose(
        [
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

        ]
    )
    def __init__(self):
        print("")

    def __getitem__(self, index):
        print ("working "+str(index))
        dataEntryPath = '../20bn-jester-v1/'+str(index)+'/00001.jpg'
        formattedImage = self.transform(Image.open(dataEntryPath))
        return (formattedImage, index)

    def __add__(self, other):
        return ConcatDataset([self, other])
    
    def __len__(self):
        return 148092
