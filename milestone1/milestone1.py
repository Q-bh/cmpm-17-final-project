import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def display_images(images, labels):
    plt.figure(figsize=(60, 40), dpi=100)
    class_labels = np.unique(labels)
    images_per_class = 15

    for class_idx, class_label in enumerate(class_labels):
        class_images = [img for img, lbl in zip(images, labels) if lbl == class_label]
        # Using 10 maximum
        class_images = class_images[:images_per_class]

        for i, img in enumerate(class_images):
            plt.subplot(len(class_labels), images_per_class, class_idx * images_per_class + i + 1)
            # Passing the PIL image to imshow directly
            plt.imshow(img, cmap='gray', vmin=0, vmax=255)
            plt.title(class_label)
            plt.axis('off')

    plt.tight_layout()
    plt.show()

# === Main code ===
dataset_path = '../dataset/train'
class_paths = glob.glob(os.path.join(dataset_path, '*'))

images = []
labels = []

for class_path in class_paths:
    class_label = os.path.basename(class_path)
    img_paths = glob.glob(os.path.join(class_path, '*.jpg'))

    if len(img_paths) > 15:
        img_paths = random.sample(img_paths, 15)

    for img_path in img_paths:
        with Image.open(img_path) as img:
            images.append(img.copy())
        labels.append(class_label)

# Keep it as a list, no need to convert it to NumPy array
display_images(images, labels)
