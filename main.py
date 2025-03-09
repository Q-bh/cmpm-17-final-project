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
    def __init__(self, num_classes = 6):
        super().__init__()

        # Two convolutional layers to avoid overfitting.
        # More layers can be added depending on the performance.

        # First convolution: Input channel = 1(Gray Scale) -> Output Channel = 16
        # Input: (batch_size, 1, 128, 128)
        # conv1: (batch_size, 16, 128, 128) [kernel_size = 3, padding = 1]
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, padding = 1)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2) # Make the image as a half size (64)

        # Second convolution: Input channel = 16 -> Output Channel = 32
        # Input: (batch_size, 16, 64, 64)
        # conv2: (batch_size, 32, 64, 64) [kernel_size = 3, padding = 1]
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1)
        self.relu = nn.ReLU() # Adding a non-linearity
    
        # After two poolings, image size is 128 -> 64 -> 32
        # Final feature map: (batch_size, 32, 32, 32)
        # Flatten = 32 * 32 * 32 = 32786 dimensions

        self.fc1 = nn.linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, input):
        # Input: (batch_size, 1, 128, 128)
        # The original data is 48*48, but we transform the image size to 128*128

        x = self.conv1(input)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.flatten(start_dim = 1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
    
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

#loss_function = nn.L1Loss()

#optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 

#
NUM_EPOCHS = 1

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