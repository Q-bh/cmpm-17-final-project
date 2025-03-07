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

# CLASSES

# Dataset
class CNN_Dataset(Dataset):
    def __init__(self, data, transform):
        self.data = datasets.ImageFolder(root=data, transform=transform)
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.data[index]

# Linear Neural Network
class CNN_Main(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        #self.linear1 = nn.Linear(..., ...)
        self.relu = nn.ReLU
    
    def forward(self, input):
        """
        partial = self.conv1(input)
        partial = self.relu(partial)
        partial = self.pool(partial)
        partial = partial.flatten(start_dim=1)
        partial = self.linear1(partial)
        """
        
    
# IMAGE TRANSFORMATIONS - Increases model robustness
train_transforms = v2.Compose([
    v2.Resize((128, 128)),     # Resizes image to 128x128; Original data is 48x48
    v2.RandomHorizontalFlip(), # Flips images horizontally with 50% probability
    v2.RandomRotation(30),     # Rotation on images up to 30 degrees
    v2.Grayscale(1),           # Images are grayscale already, but this properly makes the tensors 1 channel
    #v2.Lambda(add_noise),     # Adding noise, depending on the model performance
    v2.ToTensor(),
    v2.Normalize([0.5], [0.5]) # Normalization
])

# Only transforms for matching the size of images.
test_transforms = v2.Compose([
    v2.Resize(128, 128),
    v2.Grayscale(1),
    v2.ToTensor(),
    v2.Normalize([0.5], [0.5])
])

# DATASETS + DATALOADERS
train_dataset = CNN_Dataset("dataset/train", train_transforms)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True,)

test_dataset = CNN_Dataset("dataset/test", test_transforms)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# INITIALIZATIONS
model = CNN_Main()

# TRAINING LOOP
for i in range(NUM_EPOCHS):

    for image, label in train_dataloader:
        print(f"{image.shape}  <- SAMPLE STUFF, OUTPUT/LABELS: {label.shape}")

        # Sample code from midterm; uncomment & use when implementing real loop
        """
        # PREDICT - Pass training inputs through neural network
        pred = model(image)

        # SCORE - Higher number = Worse performance
        loss = loss_function(pred, label)

        # LEARN
        loss.backward()       # Calculates function slope
        optimizer.step()      # Updates model parameters
        optimizer.zero_grad() # Resets optimizer to be ready for next epoch
        """

# TESTING LOOP    
with torch.no_grad():

    for image, label in test_dataloader:
        print(f"<{image}>  <- SAMPLE STUFF, OUTPUT/LABELS: {label}")

        # Sample code from midterm; uncomment & use when implementing real loop
        """
        # PREDICT
        pred = model(image)

        # SCORE
        loss = loss_function(pred, label)
        """