import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.datasets as datasets
from torchvision.transforms import v2
import wandb


# Hyperparameters
dropout_rate = 0.2
conv1_out_channels = 16
conv1_kernel_size = 3
conv1_padding = 1
conv2_out_channels = 32
conv2_kernel_size = 3
conv2_padding = 1
conv3_out_channels = 64
conv3_kernel_size = 3
conv3_padding = 1
conv4_out_channels = 128
conv4_kernel_size = 3
conv4_padding = 1
batch_size = 32
fc1_units = 256
fc2_units = 128
batch_norm = True   # Batch Normalization On/Off
lr = 0.001
NUM_EPOCHS = 30
dropout = False # Dropout On/Off
scheduler = False   # Scheduler On/Off
weight_decay = False    # Weight_decay On/Off


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
    def __init__(self, num_classes=6):
        super().__init__()

        # First convolution:
        # Input: (batch_size, 1, 128, 128)
        # conv1: (batch_size, conv1_out_channels, 128, 128)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv1_out_channels, 
                            kernel_size=conv1_kernel_size, padding=conv2_kernel_size)
        self.bn1 = nn.BatchNorm2d(conv1_out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolution:
        # Input: (batch_size, conv1_out_channels, 64, 64)
        # conv2: (batch_size, conv2_out_channels, 64, 64)
        self.conv2 = nn.Conv2d(in_channels=conv1_out_channels, out_channels=conv2_out_channels, 
                            kernel_size=conv2_kernel_size, padding=conv2_padding)
        self.bn2 = nn.BatchNorm2d(conv2_out_channels)

        # Third convolution:
        # Input: (batch_size, conv2_out_channels, 32, 32)
        # conv3: (batch_size, conv3_out_channels, 32, 32)
        self.conv3 = nn.Conv2d(in_channels=conv2_out_channels, out_channels=conv3_out_channels, 
                            kernel_size=conv3_kernel_size, padding=conv3_padding)
        self.bn3 = nn.BatchNorm2d(conv3_out_channels)

        # Fourth convolution (Added):
        # Input: (batch_size, conv3_out_channels, 16, 16)
        # conv4: (batch_size, conv4_out_channels, 16, 16)
        self.conv4 = nn.Conv2d(in_channels=conv3_out_channels, out_channels=conv4_out_channels, 
                            kernel_size=conv4_kernel_size, padding=conv4_padding)
        self.bn4 = nn.BatchNorm2d(conv4_out_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        # After 4 poolings, image size: 128 -> 64 -> 32 -> 16 -> 8
        # fc1 입력 크기: conv4_out_channels * 8 * 8
        self.fc1 = nn.Linear(conv4_out_channels * 8 * 8, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, num_classes)

    def forward(self, input):
        # Input: (batch_size, 1, 128, 128)
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def main(): 
    # DEVICE SELECTION
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # wandb initialization: Saving the name of run and project
    config = {
        "dropout_rate": dropout_rate,
        "conv1_out_channels": conv1_out_channels,
        "conv1_kernel_size": conv1_kernel_size,
        "conv1_padding": conv1_padding,
        "conv2_out_channels": conv2_out_channels,
        "conv2_kernel_size": conv2_kernel_size,
        "conv2_padding": conv2_padding,
        "conv3_out_channels": conv3_out_channels,
        "conv3_kernel_size": conv3_kernel_size,
        "conv3_padding": conv3_padding,
        "conv4_out_channels": conv4_out_channels,
        "conv4_kernel_size": conv4_kernel_size,
        "conv4_padding" : conv4_padding,
        "batch_size": batch_size,
        "fc1_units": fc1_units,
        "batch_norm": True,        # Batch Normalization On/Off
        "lr": lr,
        "NUM_EPOCHS": NUM_EPOCHS,
        "scheduler": False,        # Scheduler On/Off
        "weight_decay": False      # Weight_decay On/Off
    }
    run = wandb.init(project="CMPM17_FINAL", name="Epoch 30 lr 0.001 3fc conv4 cpu", config=config)

    print("Configured hyperparameters: ")
    print(wandb.config)

    # IMAGE TRANSFORMATIONS - Increases model robustness
    train_transforms = v2.Compose([
        v2.RandomHorizontalFlip(), # Flips images horizontally with 50% probability
        v2.RandomRotation(30),     # Rotation on images up to 30 degrees
        v2.Resize((128, 128)),     # Resizes image to 128x128; Original data is 48x48
        v2.Grayscale(1),           # Images are grayscale already, but this properly makes the tensors 1 channel
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
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = CNN_Dataset("dataset/test", test_transforms)

    # Mixing test dataset indices randomly and then divided it by half
    indices = np.random.permutation(len(test_dataset))
    half = len(indices) // 2
    validation_indices = indices[:half]
    test_indices = indices[half:]

    # Dividing the test set into two, one is for the validation
    validation_dataset = Subset(test_dataset, validation_indices)
    test_dataset = Subset(test_dataset, test_indices)

    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # INITIALIZATIONS
    model = CNN_Main().to(device)

    # HOW TO READ CROSS ENTROPY LOSS:
    # For an n-class problem, randomly guessing should create an expected loss of -ln(1/n)
    # For this model, of 6 classes, it's -ln(1/6) = 1.79
    # Essentially, the model will start around 1.79 loss and should slowly go down from there
    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr) #weight_decay=0.01) ###
    #scheduler = ExpLR(optimizer, gamma=0.9)

    # TRAINING LOOP
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss_epoch = 0.0
        correct_pred = 0
        total_pred = 0

        for image, label in train_dataloader:

            image = image.to(device)
            label = label.to(device)

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
            #scheduler.step()  # With no parameter

            # Saving learning metrics in each batches on wandb
            run.log({"train_batch_losses": loss.item()})

            # SCORE - Higher number = Worse performance, it's per one batch
            print(f"Batch Loss: {loss.item():.4f}")
            print(f"Predicted: {predictions}")
            print(f"Confidences: {max_confidences}\n")
            print(f"Epoch: {epoch}")
        
        # VALIDATION LOOP
        model.eval()
        val_loss = 0.0
        val_correct_pred = 0
        val_total_pred = 0

        with torch.no_grad():
            for image, label in validation_dataloader:
                    
                image = image.to(device)
                label = label.to(device)

                # PREDICT
                pred = model(image)
                # SCORE
                loss = loss_function(pred, label)
                val_loss += loss.item() * image.size(0)
                val_total_pred += label.size(0)
                    
                confidences = torch.softmax(pred, dim=1)
                max_confidences, predictions = torch.max(confidences, dim=1)
                val_correct_pred += (predictions == label).sum().item()

                # Saving learning metrics in each batches on wandb
                run.log({"val_batch_losses": loss.item()})
    
    # Average loss and accuracy calculation after one epoch on training
    avg_loss_epoch = total_loss_epoch / total_pred
    avg_accuracy_epoch = correct_pred / total_pred
    print(f"Epoch {epoch+1} - Average Loss: {avg_loss_epoch:.4f}, Accuracy: {avg_accuracy_epoch:.4f}\n")

    # Saving learning metrics in each epoches on wandb
    run.log({"train_loss": avg_loss_epoch, "avg_train_accuracy": avg_accuracy_epoch})

    # Average loss and accuracy calculation after one epoch on validation
    avg_val_loss = val_loss / val_total_pred
    accuracy_val = val_correct_pred / val_total_pred
    print(f"Epoch {epoch+1} - Validation Average Loss: {avg_val_loss:.4f}, Accuracy: {accuracy_val:.4f}\n")

    # Saving validation metrics on wandb
    run.log({"avg_val_loss": avg_val_loss, "avg_val_accuracy": accuracy_val})

    # === Saving Model ===
    model_save_path = "cnn_model.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # # === Loading Model ===
    # # Load the saved state dict after creating new model instance
    # loaded_model = CNN_Main().to(device)
    # loaded_model.load_state_dict(torch.load(model_save_path))
    # loaded_model.eval()  # Converting the mode to evaluation
    # print(f"Model loaded from {model_save_path}")
    
    # TESTING LOOP
    model.eval()
    total_loss_test = 0.0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for image, label in test_dataloader:

            image = image.to(device)
            label = label.to(device)
            # PREDICT
            pred = model(image)

            # SCORE
            loss = loss_function(pred, label)
            total_loss_test += loss.item() * image.size(0)
            total_test += label.size(0)

            confidences = torch.softmax(pred, dim = 1)
            max_confidences, predictions = torch.max(confidences, dim = 1)
            correct_test += (predictions == label).sum().item()
    
            # Saving learning metrics in each batches on wandb
            run.log({"test_batch_loss": loss.item()})

    avg_loss_test = total_loss_test / total_test
    accuracy_test = correct_test / total_test

    print(f"Test Set - Average Loss: {avg_loss_test:.4f}, Accuracy: {accuracy_test:.4f}")

    # Saving test metrics on wandb as well
    run.log({"test_loss": avg_loss_test, "test_accuracy": accuracy_test})

    run.finish() # wandb run execution

if __name__ == "__main__":
    main()