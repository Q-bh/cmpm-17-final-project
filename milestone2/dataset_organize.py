import os
import shutil
import random

# Moving the images
source_folder = "../dataset/train/disgust"
destination_folder = "../dataset/train/angry"

source_folder2 = "../dataset/test/disgust"
destination_folder2 = "../dataset/test/angry"

# Check the existence of the folder. If not, make it
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Moving the images one by one
for filename in os.listdir(source_folder):
    file_path = os.path.join(source_folder, filename)
    if os.path.isfile(file_path):
        shutil.move(file_path, destination_folder)

# Check the existence of the folder. If not, make it
if not os.path.exists(destination_folder2):
    os.makedirs(destination_folder2)

# Moving the images one by one
for filename in os.listdir(source_folder2):
    file_path = os.path.join(source_folder2, filename)
    if os.path.isfile(file_path):
        shutil.move(file_path, destination_folder2)

# Removing 2000 images randomly from happy folder
happy_folder_train = "../dataset/train/happy"
happy_folder_test = "../dataset/test/happy"
images_train = os.listdir(happy_folder_train)
images_test = os.listdir(happy_folder_test)

# The number of images to delete
num_to_delete_train = 2000
num_to_delete_test = 500

# Choosing images to delete randomly
images_to_delete_train = random.sample(images_train, min(num_to_delete_train, len(images_train)))
images_to_delete_test = random.sample(images_test, min(num_to_delete_test, len(images_test)))

# Removing images
for image in images_to_delete_train:
    image_path = os.path.join(happy_folder_train, image)
    if os.path.isfile(image_path):
        os.remove(image_path)

for image in images_to_delete_test:
    image_path = os.path.join(happy_folder_test, image)
    if os.path.isfile(image_path):
        os.remove(image_path)


# DONE!