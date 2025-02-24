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

# IMAGE TRANSFORMATIONS
transform = v2.Compose([
    v2.ToTensor(),
])

# DATASETS + DATALOADERS
train_dataset = datasets.ImageFolder(root="dataset/train", transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True,)

test_dataset = datasets.ImageFolder(root="dataset/test", transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

for x, y in train_dataloader:
    print(x.shape, y.shape)