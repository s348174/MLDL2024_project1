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
    
class CityScapesSegmentation(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, target_transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.target_transform = target_transform

        self.images = sorted([
            os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
            if fname.endswith(".png") or fname.endswith(".jpg")
        ])
        self.labels = sorted([
            os.path.join(label_dir, fname) for fname in os.listdir(label_dir)
            if fname.endswith(".png")
        ])
        assert len(self.images) == len(self.labels), "Mismatch between images and labels"

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        label = Image.open(self.labels[index])  # Should be mode "L" with class indices

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        else:
            label = torch.from_numpy(np.array(label)).long()

        return image, label

    def __len__(self):
        return len(self.images)