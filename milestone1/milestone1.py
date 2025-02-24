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
        # 최대 10장까지만 사용
        class_images = class_images[:images_per_class]

        for i, img in enumerate(class_images):
            plt.subplot(len(class_labels), images_per_class, class_idx * images_per_class + i + 1)
            # PIL Image를 직접 imshow에 넘긴다
            plt.imshow(img, cmap='gray', vmin=0, vmax=255)
            plt.title(class_label)
            plt.axis('off')

    plt.tight_layout()
    plt.show()

# === 메인 코드 ===
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

# 리스트로 유지 (굳이 np.array로 변환할 필요 없음)
display_images(images, labels)
