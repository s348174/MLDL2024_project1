from torch.utils.data import Dataset
from PIL import Image
import os

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