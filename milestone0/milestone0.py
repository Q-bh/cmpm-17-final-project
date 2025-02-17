import os
import matplotlib.pyplot as plt

# Part1: Checking the class distribution

# Dictionary that stores the name of the class and the number of images in the class
train_distribution = {}
test_distribution = {}

# Setting up for the path of folders
folders = ['../dataset/train', '../dataset/test']
folder_names = ['Train', 'Test']

# Calculatiing the class distribution for the train and test folders
for folder_path, name in zip(folders, folder_names):
    for class_name in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_name)
        if os.path.isdir(class_path):
            if name == 'Train':
                train_distribution[class_name] = len(os.listdir(class_path))
            else:
                test_distribution[class_name] = len(os.listdir(class_path))

# Plotting the bar graph
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Class distribution in the train folder
bars_train = ax[0].bar(train_distribution.keys(), train_distribution.values(), color='blue')
ax[0].set_title('Train Class Distribution')
ax[0].set_xlabel('Classes')
ax[0].set_ylabel('Number of Samples')
ax[0].tick_params(axis='x', rotation=45)

# Adding a value on the graph
for bar in bars_train:
    yval = bar.get_height()
    ax[0].text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

# Class distribution in the test folder
bars_test = ax[1].bar(test_distribution.keys(), test_distribution.values(), color='orange')
ax[1].set_title('Test Class Distribution')
ax[1].set_xlabel('Classes')
ax[1].set_ylabel('Number of Samples')
ax[1].tick_params(axis='x', rotation=45)

# Adding a value on the graph
for bar in bars_test:
    yval = bar.get_height()
    ax[1].text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Part 2: baseline accuracy calculation

# Reset for the number of samples
total_samples = 0
happy_class_samples = 0

# Calculating the number of samples in train dataset
train_folder_path = '../dataset/train'
for class_name in os.listdir(train_folder_path):
    class_path = os.path.join(train_folder_path, class_name)
    if os.path.isdir(class_path):
        num_samples = len(os.listdir(class_path))
        total_samples += num_samples
        if class_name == 'happy':  # the number of happy class
            happy_class_samples += num_samples

# baseline accuracy calculation
baseline_accuracy = happy_class_samples / total_samples if total_samples > 0 else 0

print(f'Baseline Accuracy (Happy Class): {baseline_accuracy:.2f}')