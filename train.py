# TODO: Define here your training and validation loops.

import torch
from models.deeplabv2.deeplabv2 import ResNetMulti, get_deeplab_v2
from torch.utils.data import DataLoader
from torchvision import transforms
from cityscapes import CityScapesSegmentation
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
from torchvision.transforms import functional as TF

def convert_label_ids_to_train_ids(label_np):
    # labelId to trainId mapping
    LABEL_TO_TRAINID = {
        7: 0, 8: 1, 11: 2, 12: 3, 13: 4,
        17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
        23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
        28: 15, 31: 16, 32: 17, 33: 18
    }
    label_out = 255 * np.ones_like(label_np, dtype=np.uint8)
    for label_id, train_id in LABEL_TO_TRAINID.items():
        label_out[label_np == label_id] = train_id
    return label_out

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
def deeplab_train(dataset_path, workspace_path):
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    image_dir = dataset_path + "/images/train"
    label_dir = dataset_path + "/gtFine/train"

    input_transform = transforms.Compose([
    transforms.Resize((256, 256)), # Resize to 256x256 or smaller if needed
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    target_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.NEAREST), # Resize to 256x256 or smaller if needed
        #transforms.ToTensor(),
        #transforms.Lambda(lambda x: torch.from_numpy(np.array(x)).long())
        transforms.Lambda(lambda img: torch.from_numpy(convert_label_ids_to_train_ids(np.array(img))).long())
    ])  
    dataset = CityScapesSegmentation(
        image_dir=image_dir,
        label_dir=label_dir,
        transform=input_transform,
        target_transform=target_transform,
        
    )

    #test_path = dataset_path + "/val"
    #test_data = CityScapesSegmentation(test_path, transform=transforms.ToTensor())

    # Visualize the training data
    class_names = dataset.classes
    print(f"Class names: {class_names}")
    print(f"Number of classes:", dataset.num_classes)
    print(f"Number of training samples: {len(dataset.images)}")
    print(f"Number of labels: {len(dataset.labels)}")
    # Display the first image in the dataset
    image = Image.open(dataset.images[0])
    plt.imshow(image)
    plt.title("First image in the dataset")
    plt.axis("off")
    plt.show()
    #print(f"Number of test samples: {len(test_data)}")
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
    pretrain_path = workspace_path + "/deeplab_resnet_pretrained_imagenet.pth"
    model = get_deeplab_v2(num_classes=len(class_names), pretrain=True, pretrain_model_path=pretrain_path)
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    #criterion = torch.nn.BCEWithLogitsLoss()
    #criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(50):  # Change the number of epochs
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _, _ = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print(f"Loss: {loss.item():.4f}")
            # Save model checkpoint
            if epoch % 10 == 0:
                checkpoint_file = workspace_path + "/export/deeplabv2_epoch_{}.pth".format(epoch)
                torch.save(model.state_dict(), f"deeplabv2_epoch_{epoch}.pth")
                print(f"Model saved at epoch {epoch}")
    # Save the model
    export_path = workspace_path + "/export/deeplabv2_final.pth"
    torch.save(model.state_dict(), export_path)
    print("Model saved as deeplabv2_final.pth")

def deeplab_test(dataset_path, workspace_path, save_dir=None, num_classes=19):
    model_path = workspace_path + "/export/deeplabv2_final.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare test dataset
    image_dir = os.path.join(dataset_path, "images/val")
    label_dir = os.path.join(dataset_path, "gtFine/val")

    input_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    target_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
        transforms.Lambda(lambda img: torch.from_numpy(np.array(img)).long())
    ])

    test_dataset = CityScapesSegmentation(
        image_dir=image_dir,
        label_dir=label_dir,
        transform=input_transform,
        target_transform=target_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load model
    model = get_deeplab_v2(num_classes=num_classes, pretrain=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    print("Running inference...")

    correct_pixels = 0
    total_pixels = 0

    with torch.no_grad():
        for i, (image, label) in enumerate(test_loader):
            image, label = image.to(device), label.to(device)

            output = model(image)
            pred = torch.argmax(output.squeeze(), dim=0)

            # Flatten predictions and labels
            pred_flat = pred.view(-1)
            label_flat = label.view(-1)

            # Mask out ignored pixels (ex: 255)
            mask = label_flat != 255
            correct = (pred_flat[mask] == label_flat[mask]).sum().item()
            total = mask.sum().item()

            correct_pixels += correct
            total_pixels += total
            print(f"Processed {i + 1}/{len(test_loader)} images. Correct pixels: {correct_pixels}, Total pixels: {total_pixels}")

    # After loop:
    accuracy = correct_pixels / total_pixels
    print(f"\nPixel Accuracy: {accuracy * 100:.2f}%")