from train import deeplab_train, deeplab_test

dataset_path_alberto = "/home/alberto/Documenti/Materiale scuola Alberto/MLDL2024_project1/datasets/Cityscapes/Cityspaces"
workspace_path = "/home/alberto/Documenti/Materiale scuola Alberto/MLDL2024_project1"
deeplab_train(dataset_path_alberto, workspace_path)
model_path = workspace_path + "/export/deeplabv2_final.pth"
deeplab_test(model_path, workspace_path)