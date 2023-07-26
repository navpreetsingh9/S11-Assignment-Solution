# S11-Assignment-Solution

## Introduction

This repository contains the code for an assignment where the objective is to build the training structure and train ResNet18 on the CIFAR-10 dataset for 20 epochs. The assignment requires demonstrating adherence to a specific structure, using imported functions and classes from external files, and producing specific outputs in a Google Colab notebook.

## Repository Structure

The main repository for this assignment can be found at [main-repo-era](https://github.com/navpreetsingh9/main-repo-era). The file structure of the repository is organized as follows:

```
bash
main-repo-era/
│
├── data/                         # Folder containing the CIFAR-10 dataset
├── models/                       # Directory with model definitions (e.g., ResNet18)
│   └── resnet.py
├── utils/                        # Utility functions and classes
│   ├── dataset.py
│   ├── transforms.py
│   ├── utils.py
│   └── train.py
├── notebook_assignment.ipynb     # Google Colab notebook for the assignment
├── main.py                      # Python script for running the whole training
├── README.md                     # This README file
└── ...                           # Other project files
```

## Prerequisites

- Python 3.x
- PyTorch (1.9.0 or later)
- Torchvision
- torch-summary
- torch_lr_finder
- grad-cam
- Other common libraries (e.g., NumPy, Matplotlib)

## Clone the Repository

To get started, clone this repository using the following command:

```
bash
git clone https://github.com/navpreetsingh9/main-repo-era.git
```

## Dataset

The CIFAR-10 dataset will be automatically downloaded and prepared by the script. The training and testing data will be transformed using custom ResNet transforms.

## Model

The model used for this assignment is **ResNet18**, which is a popular deep learning architecture known for its effectiveness in various image recognition tasks.

## Hyperparameters

The hyperparameters used for training are as follows:

- Batch Size: 256
- Learning Rate: 0.03
- Weight Decay: 5e-4
- Use Scheduler: True
- End Learning Rate: 10
- Number of Epochs: 20

## Training

The model will be trained on the CIFAR-10 training dataset for the specified number of epochs. During training, the model's performance on the training and validation sets will be recorded in terms of losses and accuracy. If a scheduler is used, the learning rate will be adjusted accordingly.

## Evaluation

After training, the model's performance will be evaluated on the CIFAR-10 test dataset using the CrossEntropyLoss criterion. The test results will also be used to generate some additional visualizations, such as plotting incorrect predictions and Grad-CAM visualizations.

## Overview of Functions

### 1. `train_transforms and test_transforms`

These functions define the data augmentation transforms for the training and testing datasets. `train_transforms` apply random crop with padding and cutout augmentation, while `test_transforms` apply only normalization using the provided means and standard deviations.

### 2. `Cifar10SearchDataset`

This function creates a custom dataset for CIFAR-10 with a search space that is used for data augmentation and other transformations. It loads the CIFAR-10 dataset and applies the provided transformations to the training and testing data.

### 3. `show_images`

This function displays a random set of images from the training dataset. It takes the training data loader, class_map (dictionary mapping class indices to class names), and the number of images to display as inputs.

### 4. `print_summary`

This function prints a summary of the model architecture, showing the layers and the number of parameters. It takes the model and the input size (e.g., (1, 3, 36, 36)) as inputs.

### 5. `get_adam_optimizer`

This function initializes the Adam optimizer with the provided learning rate and weight decay. It takes the model, learning rate, and weight decay as inputs and returns the Adam optimizer.

### 6. `get_lr_finder`

This function finds the optimal learning rate by performing a learning rate range test. It takes the model, optimizer, criterion, device, training data loader, and the end learning rate as inputs.

### 7. `get_onecyclelr_scheduler`

This function gets the One Cycle Learning Rate Scheduler, which adjusts the learning rate during training. It takes the optimizer, maximum learning rate, steps per epoch, and the number of epochs as inputs.

### 8. `train`

This function trains the model on the training dataset. It takes the model, device, training data loader, optimizer, and criterion as inputs.

### 9. `test`

This function evaluates the model on the test dataset. It takes the model, device, test data loader, and criterion as inputs.

### 10. `plot_network_performance`

This function plots the network's performance in terms of training and testing losses and accuracies over the specified number of epochs. It takes the number of epochs, lists of training and testing losses, and lists of training and testing accuracies as inputs.

### 11. `get_incorrect_predictions`

This function obtains a list of misclassified images from the test dataset. It takes the model, test data loader, and device as inputs and returns a list of misclassified images.

### 12. `plot_incorrect_predictions`

This function displays a gallery of misclassified images along with their predicted and ground-truth labels. It takes the list of misclassified images, class_map (dictionary mapping class indices to class names), and the number of images to display as inputs.

### 13. `show_gradcam`

This function applies GradCAM on the misclassified images and visualizes the heatmap highlighting the regions of interest. It takes the model, list of misclassified images, class_map (dictionary mapping class indices to class names), whether CUDA is available, means, standard deviations, and the number of images to process as inputs.

## Note

Please note that this overview is a general explanation of the functions' purposes. For more specific details about each function's implementation, you can refer to the respective Python files in the repository.