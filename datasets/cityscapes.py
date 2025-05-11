from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import torch
from torchvision import transforms
import glob
from torchvision.datasets import VisionDataset

class CityScapes(Dataset):
    def __init__(self, root_dir, transform=None):
        super(CityScapes, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.classes = []

        # Scan subfolders for classes and images
        for class_name in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                self.classes.append(class_name)
                self.class_to_idx[class_name] = len(self.class_to_idx)
                for fname in sorted(os.listdir(class_path)):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(class_path, fname), self.class_to_idx[class_name]))

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.samples)
    


class CityScapesSegmentation(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, target_transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.target_transform = target_transform
        self.classes = sorted(os.listdir(label_dir))

        # Match Cityscapes file naming conventions
        image_files = glob.glob(os.path.join(image_dir, '**', '*_leftImg8bit.png'), recursive=True)
        label_files = glob.glob(os.path.join(label_dir, '**', '*_gtFine_labelTrainIds.png'), recursive=True)

        print(f"Found {len(image_files)} images and {len(label_files)} labels")

        # Map basename without suffix to full path
        image_map = {
            os.path.basename(f).replace('_leftImg8bit.png', ''): f for f in image_files
        }
        label_map = {
            os.path.basename(f).replace('_gtFine_labelTrainIds.png', ''): f for f in label_files
        }

        print(f"Image map:", len(image_map.keys()))

        # Match based on shared keys
        common_keys = sorted(set(image_map.keys()) & set(label_map.keys()))
        self.images = [image_map[k] for k in common_keys]
        self.labels = [label_map[k] for k in common_keys]

        print(f"Matched {len(self.images)} image-label pairs")

        assert len(self.images) == len(self.labels), "Image-label count mismatch"

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        label = Image.open(self.labels[index])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        else:
            label = torch.from_numpy(np.array(label)).long()

        return image, label

    def __len__(self):
        return len(self.images)
