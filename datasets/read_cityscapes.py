import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from torchvision import transforms
from torchvision import datasets #nuovo, aggiunto per visionare i dati in cityscapes

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import time
import os
from PIL import Image
from tempfile import TemporaryDirectory

cudnn.benchmark = True
plt.ion()   # interactive mode

# Setup training data #NON FUNZIONA
train_data = datasets.ImageFolder(
    #root="C:/Users/marti/OneDrive/Desktop/HW Masone/Cityscapes/Cityscapes/Cityscapes/images/train/",
    root="/home/alberto/Documenti/Materiale scuola Alberto/MLDL2024_project1/datasets/Cityscapes/Cityspaces/images/train",
    transform=ToTensor()
)

# Setup test data
test_data = datasets.ImageFolder(
    root="/home/alberto/Documenti/Materiale scuola Alberto/MLDL2024_project1/datasets/Cityscapes/Cityspaces/images/val",
    transform=ToTensor()
)

# Let's check the first training sample
dataset_path = "/home/alberto/Documenti/Materiale scuola Alberto/MLDL2024_project1/datasets/Cityscapes/Cityspaces/images"
train_path = dataset_path + "/train"
dataset = datasets.ImageFolder(root=train_path, transform=transforms.ToTensor())
class_names = dataset.classes
print(f"Class names: {class_names}")
print(f"Number of classes: {len(class_names)}")
print(f"Number of training samples: {len(dataset)}")
print(f"Number of test samples: {len(test_data)}")
print(f"Image shape: {dataset[0][0].shape}")
# Let's visualize the first training sample
image, label = dataset[0]
plt.imshow(image.permute(1, 2, 0))
plt.title(f"Label: {class_names[label]}")
plt.axis("off")
plt.show()
# Let's visualize the first test sample
image, label = test_data[0]
plt.imshow(image.permute(1, 2, 0))
plt.title(f"Label: {class_names[label]}")
plt.axis("off")
plt.show()