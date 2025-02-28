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
transform = v2.Compose([
    v2.ToTensor(),
    v2.Grayscale(1), # Images are grayscale already, but this properly makes the tensors 1 channel
])

# DATASETS + DATALOADERS
train_dataset = CNN_Dataset("dataset/train")
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True,)

test_dataset = CNN_Dataset("dataset/test")
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# INITIALIZATIONS
#model = NN()

#loss_function = nn.L1Loss()

#optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 

#
NUM_EPOCHS = 1

# TRAINING LOOP
for i in range(NUM_EPOCHS):

    for image, label in train_dataloader:
        print(f"<{image}>  <- SAMPLE STUFF, OUTPUT/LABELS: {label}")

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