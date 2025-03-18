# Excess libraries are included in case they are needed
import optuna
import time
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.datasets as datasets
from torchvision.transforms import v2
import wandb
from torch.optim.lr_scheduler import ExponentialLR as ExpLR

# Dataset
class CNN_Dataset(Dataset):
    def __init__(self, data, transform):
        self.data = datasets.ImageFolder(root=data, transform=transform)
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.data[index]

# DeepCNN_Main 모델 (예시로, 깊은 모델 variant)
class DeepCNN_Main(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.5, 
                conv1_channels=16, conv2_channels=32, conv3_channels=64, fc1_units=128, 
                conv1_kernel_size=3, conv1_padding_size=1, 
                pool_kernel_size=2, pool_stride_size=2, 
                conv2_kernel_size=3, conv2_padding_size=1,
                conv3_kernel_size=3, conv3_padding_size=1,
                activation=nn.ReLU(), input_size=128, transform=None):
        
        super().__init__()

        self.conv1 = nn.Conv2d(1, conv1_channels, kernel_size=conv1_kernel_size, padding=conv1_padding_size)
        self.bn1 = nn.BatchNorm2d(conv1_channels)
        self.pool = nn.MaxPool2d(pool_kernel_size, pool_stride_size)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size=conv2_kernel_size, padding=conv2_padding_size)
        self.bn2 = nn.BatchNorm2d(conv2_channels)
        self.conv3 = nn.Conv2d(conv2_channels, conv3_channels, kernel_size=conv3_kernel_size, padding=conv3_padding_size)
        self.bn3 = nn.BatchNorm2d(conv3_channels)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)

        # fc1 입력 크기 계산을 위한 dummy 입력
        from torchvision.transforms import v2
        transform = v2.Compose([
            v2.Resize((input_size, input_size)),
            v2.ToTensor(),
            v2.Normalize([0.5], [0.5])
        ])
        dummy_img = Image.new('L', (input_size, input_size))
        dummy_tensor = transform(dummy_img).unsqueeze(0)

        with torch.no_grad():
            x = self.conv1(dummy_tensor)
            x = self.bn1(x)
            x = self.activation(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.activation(x)
            x = self.pool(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.activation(x)
            x = self.pool(x)
            flattened_size = x.view(1, -1).size(1)
        print("Calculated flattened size:", flattened_size)
        self.fc1 = nn.Linear(flattened_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, num_classes)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.pool(x)
        
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def objective(trial):
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Hyperparameter 공간 설정
    param_space = {
        'learning_rate': ('float', 0.0001, 0.01, 'log'),
        'dropout_rate': ('float', 0.3, 0.7, None),
        'conv1_channels': ('int', 16, 32, None),
        'conv2_channels': ('int', 32, 64, None),
        'conv3_channels': ('int', 64, 128, None),
        'fc1_units': ('int', 128, 256, None),
        'batch_size': ('categorical', [16, 32, 64], None, None),
        'conv1_kernel_size': ('categorical', [3, 5, 7], None, None),
        'conv1_padding_size': ('int', 1, 3, None),
        'pool_kernel_size': ('categorical', [2, 3], None, None),
        'pool_stride_size': ('categorical', [2, 3], None, None),
        'conv2_kernel_size': ('categorical', [3, 5], None, None),
        'conv2_padding_size': ('int', 1, 2, None),
        'conv3_kernel_size': ('categorical', [3, 5, 7], None, None),
        'conv3_padding_size': ('int', 1, 3, None),
        'weight_decay': ('float', 0.0, 1e-3, None)
    }

    suggested_params = {}
    for param, (ptype, low, high, extra) in param_space.items():
        if ptype == 'int':
            suggested_params[param] = trial.suggest_int(param, low, high)
        elif ptype == 'float':
            if extra == 'log':
                suggested_params[param] = trial.suggest_float(param, low, high, log=True)
            else:
                suggested_params[param] = trial.suggest_float(param, low, high)
        elif ptype == 'categorical':
            suggested_params[param] = trial.suggest_categorical(param, low)

    print("Suggested parameters:", suggested_params)

    run = wandb.init(project="CMPM17_FINAL_Optuna", name="March 14", reinit=True)

    train_transforms = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(30, expand=False, fill=0),
        v2.Resize((128, 128)),
        v2.Grayscale(1),
        v2.ToTensor(),
        v2.Normalize([0.5], [0.5])
    ])

    test_transforms = v2.Compose([
        v2.Grayscale(1),
        v2.Resize((128, 128)),
        v2.ToTensor(),
        v2.Normalize([0.5], [0.5])
    ])

    train_dataset = CNN_Dataset("dataset/train", train_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=suggested_params['batch_size'], shuffle=True)

    test_dataset = CNN_Dataset("dataset/test", test_transforms)
    indices = np.random.permutation(len(test_dataset))
    half = len(indices) // 2
    validation_indices = indices[:half]
    test_indices = indices[half:]
    validation_dataset = Subset(test_dataset, validation_indices)
    test_dataset = Subset(test_dataset, test_indices)
    validation_dataloader = DataLoader(validation_dataset, batch_size=suggested_params['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=suggested_params['batch_size'], shuffle=True)

    model = DeepCNN_Main(num_classes=6,
                        dropout_rate=suggested_params['dropout_rate'],
                        conv1_channels=suggested_params['conv1_channels'],
                        conv2_channels=suggested_params['conv2_channels'],
                        conv3_channels=suggested_params['conv3_channels'],
                        fc1_units=suggested_params['fc1_units'],
                        conv1_kernel_size=suggested_params['conv1_kernel_size'],
                        conv1_padding_size=suggested_params['conv1_padding_size'],
                        pool_kernel_size=suggested_params['pool_kernel_size'],
                        pool_stride_size=suggested_params['pool_stride_size'],
                        conv2_kernel_size=suggested_params['conv2_kernel_size'],
                        conv2_padding_size=suggested_params['conv2_padding_size'],
                        conv3_kernel_size=suggested_params['conv3_kernel_size'],
                        conv3_padding_size=suggested_params['conv3_padding_size'],
                        input_size=128,
                        transform=train_transforms).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=suggested_params['learning_rate'], 
                                weight_decay=suggested_params['weight_decay'])
    scheduler = ExpLR(optimizer, gamma=0.9)

    NUM_EPOCHS = 30
    best_val_loss = float('inf')
    early_stop_counter = 0
    patience = 3
    best_model_state = None
    best_model_path = "best_model.pt"  # 최적 모델 저장 경로 정의

    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss_epoch = 0.0
        correct_pred = 0
        total_pred = 0

        for image, label in train_dataloader:
            image = image.to(device)
            label = label.to(device)
            pred = model(image)
            loss = loss_function(pred, label)
            confidences = torch.softmax(pred, dim=1) 
            _, predictions = torch.max(confidences, dim=1)
            total_loss_epoch += loss.item() * image.size(0)
            total_pred += label.size(0)
            correct_pred += (predictions == label).sum().item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()  # epoch 인자 없이 호출
            run.log({"batch_loss": loss.item()})
            print(f"Batch Loss: {loss.item():.4f}")
            print(f"Predicted: {predictions}")
            print(f"Confidences: {confidences}\n")
        
        avg_loss_epoch = total_loss_epoch / total_pred
        accuracy_epoch = correct_pred / total_pred
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss_epoch:.4f}, Accuracy: {accuracy_epoch:.4f}\n")
        run.log({"epoch": epoch+1, "train_loss": avg_loss_epoch, "train_accuracy": accuracy_epoch})

        model.eval()
        val_loss = 0.0
        val_correct_pred = 0
        val_total_pred = 0

        with torch.no_grad():
            for image, label in validation_dataloader:
                image = image.to(device)
                label = label.to(device)
                pred = model(image)
                loss = loss_function(pred, label)
                val_loss += loss.item() * image.size(0)
                val_total_pred += label.size(0)
                confidences = torch.softmax(pred, dim=1)
                _, predictions = torch.max(confidences, dim=1)
                val_correct_pred += (predictions == label).sum().item()

        avg_val_loss = val_loss / val_total_pred
        accuracy_val = val_correct_pred / val_total_pred
        print(f"Epoch {epoch+1} - Validation Average Loss: {avg_val_loss:.4f}, Accuracy: {accuracy_val:.4f}\n")
        run.log({"epoch": epoch+1, "avg_val_loss": avg_val_loss, "avg_val_accuracy": accuracy_val})
        scheduler.step(avg_val_loss)

        # Early Stopping 및 최적 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            best_model_state = model.state_dict().copy()
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with validation loss: {best_val_loss:.4f}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        model.train()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Restored best model for final evaluation")

    run.finish()

    model.eval()
    final_val_loss = 0.0
    final_val_correct = 0
    final_val_total = 0

    with torch.no_grad():
        for image, label in validation_dataloader:
            image = image.to(device)
            label = label.to(device)
            pred = model(image)
            loss = loss_function(pred, label)
            final_val_loss += loss.item() * image.size(0)
            final_val_total += label.size(0)
            confidences = torch.softmax(pred, dim=1)
            _, predictions = torch.max(confidences, dim=1)
            final_val_correct += (predictions == label).sum().item()

    final_avg_val_loss = final_val_loss / final_val_total
    final_accuracy_val = final_val_correct / final_val_total
    print(f"Final Validation - Average Loss: {final_avg_val_loss:.4f}, Accuracy: {final_accuracy_val:.4f}\n")
    print("Suggested parameters:", suggested_params)

    return final_avg_val_loss

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    
    print("Best trial:")
    trial = study.best_trial
    print("Validation Loss:", trial.value)
    print("Best hyperparameters:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
