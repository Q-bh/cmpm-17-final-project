# Excess libraries are included in case they are needed
# They will be removed by the final project's completion if they remain unused
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
from torchvision.transforms import v2

# Dataset
class CNN_Dataset(Dataset):
    def __init__(self, data):
        self.data = datasets.ImageFolder(root=data, transform=transform)
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.data[index]
    
# IMAGE TRANSFORMATIONS
train_transforms = v2.Compose([
    v2.RandomResizedCrop(224),
    v2.RandomHorizontalFlip(),
    v2.RandomRotation(30),
    v2.Grayscale(1), # Images are grayscale already, but this properly makes the tensors 1 channel
    v2.ToTensor(),
    v2.Normalize([0.5], [0.5])
])

test_transforms = v2.Compose([
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.Grayscale(1),
    v2.ToTensor(),
    v2.Normalize([0.5], [0.5])
])

# DATASETS + DATALOADERS
train_dataset = CNN_Dataset("dataset/train")
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True,)

test_dataset = CNN_Dataset("dataset/test")
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

for x, y in train_dataloader:
    print(x.shape, y.shape)