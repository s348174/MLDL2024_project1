from train import deeplab_train, deeplab_test, bisenet_test
from utils import compute_class_weights
import os

dataset_path_alberto = "/home/alberto/Documenti/Materiale scuola Alberto/MLDL2024_project1/datasets/Cityscapes/Cityspaces"
#dataset_path_emanuele = "C:/Users/marti/OneDrive/Desktop/HW_Masone/MLDL2024_project1/datasets/Cityscapes/Cityscapes/Cityscapes"
workspace_path = "/home/alberto/Documenti/Materiale scuola Alberto/MLDL2024_project1"
pretrained_image_path = "/home/alberto/Documenti/Materiale scuola Alberto/MLDL2024_project1/deeplab_resnet_pretrained_imagenet.pth"
num_epochs = 1
#deeplab_train(dataset_path_alberto, workspace_path, pretrained_image_path, num_epochs)

model_path = workspace_path + "/export/bisenet_final_balanced.pth"
#deeplab_test(dataset_path_emanuele, model_path)
bisenet_test(dataset_path_alberto, model_path)



# Compute class weights
"""label_dir_emanuele = "C:/Users/marti/OneDrive/Desktop/HW_Masone/MLDL2024_project1/datasets/Cityscapes/Cityscapes/Cityscapes/gtFine/train"
class_frequencies = compute_class_weights(dataset_path_emanuele + "/gtFine/train", num_classes=19)
for class_idx, freq in class_frequencies.items():
    print(f"Metric {class_idx}:")
    for i in range(len(freq)):
        print(f"Class {i}: {freq[i]}; ")
    print("\n")"""


