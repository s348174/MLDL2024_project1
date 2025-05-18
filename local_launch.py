from train import deeplab_train, deeplab_test
from utils import compute_class_weights

dataset_path_alberto = "/home/alberto/Documenti/Materiale scuola Alberto/MLDL2024_project1/datasets/Cityscapes/Cityspaces"
#dataset_path_emanuele = "C:/Users/marti/OneDrive/Desktop/HW Masone/MLDL2024_project1/datasets/Cityscapes/Cityscapes"
workspace_path = "/home/alberto/Documenti/Materiale scuola Alberto/MLDL2024_project1"
pretrained_image_path = "/home/alberto/Documenti/Materiale scuola Alberto/MLDL2024_project1/export/deeplabv2_epoch_22.pth"
num_epochs = 1
#deeplab_train(dataset_path_alberto, workspace_path, pretrained_image_path, num_epochs)
model_path = workspace_path + "/export/deeplabv2_epoch_8.pth"
deeplab_test(dataset_path_alberto, model_path)

"""
class_frequencies = compute_class_weights(dataset_path_alberto + "/gtFine/val", num_classes=19)
for class_idx, freq in class_frequencies.items():
    print(f"Metric {class_idx}:")
    for i in range(len(freq)):
        print(f"Class {i}: {freq[i]}; ")
    print("\n")
    """