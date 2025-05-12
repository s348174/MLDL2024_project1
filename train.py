# TODO: Define here your training and validation loops.

import torch
from models.deeplabv2.deeplabv2 import ResNetMulti, get_deeplab_v2
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.cityscapes import CityScapesSegmentation
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory

# Setup training data
# Assuming the dataset is structured as follows:
# /path/to/train/class1/image1.png
# /path/to/train/class1/image2.png
# /path/to/train/class2/image1.png
# /path/to/train/class2/image2.png
#dataset_path = "/home/alberto/Documenti/Materiale scuola Alberto/MLDL2024_project1/datasets/Cityscapes/Cityspaces/images"
#train_path = dataset_path + "/train"
#image_dir = "/home/alberto/Documenti/Materiale scuola Alberto/MLDL2024_project1/datasets/Cityscapes/Cityspaces/images/train"
#label_dir = "/home/alberto/Documenti/Materiale scuola Alberto/MLDL2024_project1/datasets/Cityscapes/Cityspaces/labels/train"
def deeplab_train(dataset_path, pretrain_path):
    image_dir = dataset_path + "/images/train"
    label_dir = dataset_path + "/gtFine/train"

    dataset = CityScapesSegmentation(
        image_dir=image_dir,
        label_dir=label_dir,
        transform=transforms.ToTensor(),
        target_transform=transforms.Lambda(lambda x: torch.from_numpy(np.array(x)).long())
    )

    #test_path = dataset_path + "/val"
    #test_data = CityScapesSegmentation(test_path, transform=transforms.ToTensor())

    # Visualize the training data
    class_names = dataset.classes
    print(f"Class names: {class_names}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Number of training samples: {len(dataset.images)}")
    print(f"Number of labels: {len(dataset.labels)}")
    print(f"First image", dataset.images[0])
    #print(f"Number of test samples: {len(test_data)}")
    # Let's visualize the first training sample
    image = Image.open(dataset.images[0])
    #plt.imshow(image.permute(1, 2, 0))
    #plt.title(f"Label: {class_names[image]}")
    plt.axis("off")
    plt.show()
    # Let's visualize the first test sample
    #image, label = test_data[0]
    #plt.imshow(image.permute(1, 2, 0))
    #plt.title(f"Label: {class_names[label]}")
    #plt.axis("off")
    #plt.show()

    # Define the loader
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)

    # Prepare model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_deeplab_v2(num_classes=len(class_names), pretrain=True, pretrain_model_path=pretrain_path)
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop (1 epoch example)
    for epoch in range(1):  # Change the number of epochs
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _, _ = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Loss: {loss.item():.4f}")
            # Save model checkpoint
            if epoch % 10 == 0:
                torch.save(model.state_dict(), f"deeplabv2_epoch_{epoch}.pth")
                print(f"Model saved at epoch {epoch}")
    # Save the model
    torch.save(model.state_dict(), "deeplabv2_final.pth")
    print("Model saved as deeplabv2_final.pth")