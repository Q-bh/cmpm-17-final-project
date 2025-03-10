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

        # First convolution:    
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

        self.fc1 = nn.Linear(32 * 32 * 32, 128)
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
    v2.Resize((128, 128)),
    v2.Grayscale(1),
    v2.ToTensor(),
    v2.Normalize([0.5], [0.5])
])

# DATASETS + DATALOADERS
train_dataset = CNN_Dataset("dataset/train", train_transforms)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = CNN_Dataset("dataset/test", test_transforms)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# INITIALIZATIONS
model = CNN_Main()

# HOW TO READ CROSS ENTROPY LOSS:
# For an n-class problem, randomly guessing should create an expected loss of -ln(1/n)
# For this model, of 6 classes, it's -ln(1/6) = 1.79
# Essentially, the model will start around 1.79 loss and should slowly go down from there
loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 

#
NUM_EPOCHS = 3

# TRAINING LOOP
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss_epoch = 0.0
    correct_pred = 0
    total_pred = 0

    for image, label in train_dataloader:
        # PREDICT - Pass training inputs through neural network
        pred = model(image)

        # Loss calculation with CrossEntropy
        loss = loss_function(pred, label)

        # torch.softmax() - Convert predictions into confidences
        confidences = torch.softmax(pred, dim=1) 

        # torch.max() - Get most confident class and its probability
        max_confidences, predictions = torch.max(confidences, dim=1)

        # Loss for epochs and batches
        total_loss_epoch += loss.item() * image.size(0) # loss for one epoch
        total_pred += label.size(0) # Same as the batch size
        correct_pred += (predictions == label).sum().item() # How many images did the model predict correctly

        # LEARN
        loss.backward()       # Calculates function slope
        optimizer.step()      # Updates model parameters
        optimizer.zero_grad() # Resets optimizer to be ready for next epoch

        # SCORE - Higher number = Worse performance, it's per one batch
        print(f"Batch Loss: {loss.item():.4f}")
        print(f"Predicted: {predictions}")
        print(f"Confidences: {max_confidences}\n")
        
    # Average loss and accuracy calculation after one epoch
    avg_loss_epoch = total_loss_epoch / total_pred
    accuracy_epoch = correct_pred / total_pred
    print(f"Epoch {epoch+1} - Average Loss: {avg_loss_epoch:.4f}, Accuracy: {accuracy_epoch:.4f}\n")

# TESTING LOOP
# We commented the test loop because we don't want to run it until we ACTUALLY want to test the model.
# model.eval()
# total_loss_test = 0.0
# correct_test = 0
# total_test = 0

# with torch.no_grad():

#     for image, label in test_dataloader:
#         # PREDICT
#         pred = model(image)

#         # SCORE
#         loss = loss_function(pred, label)
#         total_loss_test += loss.item() * image.size(0)
#         total_test += label.size()

#         confidences = torch.softmax(pred, dim = 1)
#         max_confidences, predictions = torch.max(confidences, dim = 1)
#         correct_test += (predictions == label).sum().item()

# avg_loss_test = total_loss_test / total_test
# accuracy_test = correct_test / total_test

# print(f"Test Set - Average Loss: {avg_loss_test:.4f}, Accuracy: {accuracy_test:.4f}")