from PIL import Image
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

# TODO: implement here your custom dataset class for GTA5


class GTA5(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, target_transform=None):
        super(GTA5, self).__init__()
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.target_transform = target_transform
        self.classes = [
            'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'light',
            'sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
            'truck', 'bus', 'train', 'motorcycle', 'bicycle'
        ]
        self.num_classes = len(self.classes)

        # Find all images and labels
        image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')) + glob.glob(os.path.join(image_dir, '*.jpg')))
        label_files = sorted(glob.glob(os.path.join(label_dir, '*.png')) + glob.glob(os.path.join(label_dir, '*.jpg')))

        print(f"Found {len(image_files)} images and {len(label_files)} labels")
        
        # Match by filename (without extension)
        image_map = {os.path.splitext(os.path.basename(f))[0]: f for f in image_files}
        label_map = {os.path.splitext(os.path.basename(f))[0]: f for f in label_files}
        common_keys = sorted(set(image_map.keys()) & set(label_map.keys()))
        self.images = [image_map[k] for k in common_keys]
        self.labels = [label_map[k] for k in common_keys]

        print(f"Matched {len(self.images)} image-label pairs")

        assert len(self.images) == len(self.labels), "Image-label count mismatch"

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        label = Image.open(self.labels[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        else:
            label = torch.from_numpy(np.array(label)).long()
        return image, label

    def __len__(self):
        return len(self.images)
