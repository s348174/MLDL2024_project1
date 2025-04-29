# TODO: Define here your training and validation loops.

from torchvision import transforms
from datasets.read_cityscapes import CityScapes
# Setup training data
# Assuming the dataset is structured as follows:
# /path/to/train/class1/image1.png
# /path/to/train/class1/image2.png
# /path/to/train/class2/image1.png
# /path/to/train/class2/image2.png


dataset_path = "/home/alberto/Documenti/Materiale scuola Alberto/MLDL2024_project1/datasets/Cityscapes/Cityspaces/images"
train_path = dataset_path + "/train"
dataset = CityScapes(train_path, transform=transforms.ToTensor())
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