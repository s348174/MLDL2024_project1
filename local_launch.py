from train import deeplab_train

dataset_path_alberto = "/home/alberto/Documenti/Materiale scuola Alberto/MLDL2024_project1/datasets/Cityscapes/Cityspaces"
pretrain_path = "/home/alberto/Documenti/Materiale scuola Alberto/MLDL2024_project1/deepla_resnet_pretrained_imagenet.pth"
deeplab_train(dataset_path_alberto, pretrain_path)