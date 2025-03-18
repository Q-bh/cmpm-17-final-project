# CMPM 17 Final Project

Created by Liam Youm & Kevin Nguyen.
Liam Youm: https://github.com/Liamtsy
Kevin Nguyen: https://github.com/Q-bh

## Brief

This is the final project assignment for the UCSC Winter 2025 offering of CMPM 17-02: Intro to Machine Learning. The goal is to create, train, and test an extension of a Neural Network using real-world data.

## Description

### Dataset

For our project, we decided to use the [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) dataset from Kaggle. This dataset is widely used in the field of facial expression recognition and consists of 48×48 pixel grayscale images of faces. The images have been pre-processed using automatic registration, ensuring that each face is more or less centered and occupies approximately the same area in the image, which helps reduce variability during feature extraction.

The dataset is split into two main subsets: a training set and a test set. Both subsets include examples labeled with one of the seven emotion classes: surprise, sad, neutral, happy, fear, disgust, and angry. Specifically, the training set contains 28,709 examples, and the test set contains 3,589 examples. In our project, the input will be these facial images, and the output will be the predicted emotion for each image. The Convolutional Neural Network (CNN) we develop will output confidence scores (probabilities) for each of the seven emotion classes. Since each image belongs exclusively to one of these categories, this task is formulated as a multi-class classification problem.

### Model Architecture

Our model will be a CNN. We already have roughly over 32,000 images as inputs and outputs with a likelihood of the class that the image would belong to. The specific model architecture is TBD, and needs to be addressed on later milestones.

### Accuracy Prediction

To estimate accuracy, we first decided to establish a baseline. By analyzing the class distribution in the dataset, we determined that a simple majority classifier would serve as our baseline metric.

While there is no significant difference between the train and test datasets in terms of distribution, we observed class imbalance. Specifically, in the training dataset, the happy class accounts for 25% of the total data.

This means that if we were to use a simple majority classifier, always predicting the happy class, we would achieve 25% accuracy. With this in mind, we set our initial target accuracy at 40%, aiming for a 15 percentage point improvement over the baseline.

However, this is only an initial goal. Through continuous improvements and model optimization, our final objective is to achieve an accuracy of at least 60%.
