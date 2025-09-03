from train import deeplab_train, deeplab_test, bisenet_test, bisenet_on_gta
from utils import compute_class_weights, convert_weights_format
import os
import torch

dataset_path_alberto = "/home/alberto/Documenti/Materiale scuola Alberto/MLDL2024_project1/datasets/Cityscapes/Cityspaces"
#dataset_path_emanuele = "C:/Users/marti/OneDrive/Desktop/HW_Masone/MLDL2024_project1/datasets/Cityscapes/Cityspaces"
workspace_path = "/home/alberto/Documenti/Materiale scuola Alberto/MLDL2024_project1"
#pretrained_image_path_alberto = "/home/alberto/Documenti/Materiale scuola Alberto/MLDL2024_project1/deeplab_resnet_pretrained_imagenet.pth"
num_epochs = 1
#deeplab_train(dataset_path_alberto, workspace_path, pretrained_image_path, num_epochs)

model_path_alberto = workspace_path + "/export/bisenet_adversarial_epoch_42_10111.pth"
#model_path_emanuele = "C:/Users/marti/OneDrive/Desktop/HW_Masone/MLDL2024_project1/bisenet_on_gta_final_00001_new.pth"
    #augmentation = "wxyza" w=rotate, x=multiply, y=blur, z=flip, a=color_jitter (1 yes, 0 no)


#deeplab_test(dataset_path_emanuele, model_path)
#convert_weights_format(model_path, 50, 2, False, context_path='resnet18')
bisenet_test(dataset_path_alberto, model_path_alberto)
#bisenet_test(dataset_path_emanuele, model_path_emanuele)

#gta_path = "/home/alberto/Documenti/Materiale scuola Alberto/MLDL2024_project1/datasets/GTA5"
#gta_model = "/home/alberto/Documenti/Materiale scuola Alberto/MLDL2024_project1/export/bisenet_on_gta_final_4_batches_balanced_polylr.pth"
#bisenet_on_gta(gta_path, workspace_path, num_epochs)
#bisenet_test(dataset_path_alberto, gta_model)



# Compute class weights
"""label_dir_emanuele = "C:/Users/marti/OneDrive/Desktop/HW_Masone/MLDL2024_project1/datasets/Cityscapes/Cityscapes/Cityscapes/gtFine/train"
class_frequencies = compute_class_weights(dataset_path_emanuele + "/gtFine/train", num_classes=19)
for class_idx, freq in class_frequencies.items():
    print(f"Metric {class_idx}:")
    for i in range(len(freq)):
        print(f"Class {i}: {freq[i]}; ")
    print("\n")"""


