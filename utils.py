import numpy as np
import torch

def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=300, power=0.9):
    """Polynomial decay of learning rate
            :param init_lr is base learning rate
            :param iter is a current iteration
            :param lr_decay_iter how frequently decay occurs, default is 1
            :param max_iter is number of maximum iterations
            :param power is a polymomial power

    """
    # if iter % lr_decay_iter or iter > max_iter:
    # 	return optimizer

    lr = init_lr*(1 - iter/max_iter)**power #polynomial decay of the learning rate
    optimizer.param_groups[0]['lr'] = lr
    return lr
    # return lr


def fast_hist(a, b, n): # Confusion matrix
    '''
    a and b are label and prediction respectively
    n is the number of classes
    '''
    if isinstance(a, torch.Tensor):
        a = a.cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.cpu().numpy()
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iou(hist): # IoU metric
    epsilon = 1e-5
    return (np.diag(hist)) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)

import numpy as np
import os
from PIL import Image
from tqdm import tqdm

############
# CITYSCAPES
############

def convert_label_ids_to_train_ids(label_np):
    # labelId to trainId mapping
    LABEL_TO_TRAINID = {
        7: 0, 8: 1, 11: 2, 12: 3, 13: 4,
        17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
        23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
        28: 15, 31: 16, 32: 17, 33: 18
    }
    label_out = 255 * np.ones_like(label_np, dtype=np.uint8) # 255 is the class that has to be ignored 
    for label_id, train_id in LABEL_TO_TRAINID.items(): # Each pixel gets the train_label 
        label_out[label_np == label_id] = train_id
    return label_out

def compute_class_weights(label_dir, num_classes=19):
    """
    Computes class pixel frequencies and balancing weights from Cityscapes-style ground truth.

    Args:
        label_dir (str): Root directory of ground truth labels (e.g. gtFine/train).
        num_classes (int): Number of valid training classes (typically 19 for Cityscapes).

    Returns:
        dict with:
            - raw_counts
            - freqs
            - inv_freqs
            - median_freq_balanced
    """
    class_counts = np.zeros(num_classes, dtype=np.int64)
    
    # Check if label_dir exists
    if not os.path.exists(label_dir):
        raise ValueError(f"Label directory {label_dir} does not exist.")
    
    
    # Walk recursively through label_dir
    label_paths = []
    for root, _, files in os.walk(label_dir):
        for file in files:
            if file.endswith("_labelTrainIds.png"):
                label_paths.append(os.path.join(root, file))

    for label_path in tqdm(label_paths, desc="Computing class frequencies", disable=True):
        label = np.array(Image.open(label_path))
        #label = convert_label_ids_to_train_ids(label)
        for class_id in range(num_classes):
            class_counts[class_id] += np.sum(label == class_id)

    # Avoid divide-by-zero in any class
    class_counts[class_counts == 0] = 1

    # Compute per-class frequency
    total_pixels = np.sum(class_counts)
    freqs = class_counts / total_pixels
    inv_freqs = 1.0 / freqs

    # Median frequency balancing
    class_frequencies = class_counts / class_counts.sum()
    median = np.median(class_frequencies)
    median_freq_balanced = median / class_frequencies

    return {
        'raw_counts': class_counts,
        'freqs': freqs,
        'inv_freqs': inv_freqs,
        'median_freq_balanced': median_freq_balanced,
    }

######
# GTA5
######

# GTA5 color to trainId mapping (Cityscapes order)
GTA5_COLOR_TO_TRAINID = {
    (128, 64, 128): 0,    # road
    (244, 35, 232): 1,    # sidewalk
    (70, 70, 70): 2,      # building
    (102, 102, 156): 3,   # wall
    (190, 153, 153): 4,   # fence
    (153, 153, 153): 5,   # pole
    (250, 170, 30): 6,    # light
    (220, 220, 0): 7,     # sign
    (107, 142, 35): 8,    # vegetation
    (152, 251, 152): 9,   # terrain
    (70, 130, 180): 10,   # sky
    (220, 20, 60): 11,    # person
    (255, 0, 0): 12,      # rider
    (0, 0, 142): 13,      # car
    (0, 0, 70): 14,       # truck
    (0, 60, 100): 15,     # bus
    (0, 80, 100): 16,     # train
    (0, 0, 230): 17,      # motorcycle
    (119, 11, 32): 18,    # bicycle
}

def convert_gta5_rgb_to_trainid(label_img):
    """
    Convert a GTA5 RGB label image to Cityscapes-compatible train IDs.
    Args:
        label_img: np.ndarray (H, W, 3) or PIL.Image
    Returns:
        label_id: np.ndarray (H, W) with values in 0-18 or 255 (ignore)
    """
    if not isinstance(label_img, np.ndarray):
        label_img = np.array(label_img)
    h, w, _ = label_img.shape
    # Initialize label_id with 255, which represents the ignored class
    label_id = 255 * np.ones((h, w), dtype=np.uint8)
    for color, train_id in GTA5_COLOR_TO_TRAINID.items():
        mask = np.all(label_img == color, axis=-1)
        label_id[mask] = train_id
    return label_id

# Compute class weights for GTA5 dataset
def compute_gta5_class_weights(label_dir, num_classes=19):
    """
    Computes class pixel frequencies and balancing weights for GTA5 dataset.
    Args:
        label_dir (str): Directory containing all GTA5 label images (RGB).
        num_classes (int): Number of valid training classes.
    Returns:
        dict with:
            - raw_counts
            - freqs
            - inv_freqs
            - median_freq_balanced
    """
    class_counts = np.zeros(num_classes, dtype=np.int64)
    label_paths = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.png')]

    if len(label_paths) == 0:
        raise ValueError(f"No label files found in {label_dir}")

    for label_path in tqdm(label_paths, desc="Computing GTA5 class frequencies"):
        label_img = Image.open(label_path).convert('RGB')  # Ensure it's in RGB format
        label_np = convert_gta5_rgb_to_trainid(label_img)
        for class_id in range(num_classes):
            class_counts[class_id] += np.sum(label_np == class_id)

    # Avoid divide-by-zero
    class_counts[class_counts == 0] = 1

    total_pixels = np.sum(class_counts)
    freqs = class_counts / total_pixels
    inv_freqs = 1.0 / freqs
    class_frequencies = class_counts / class_counts.sum()
    median = np.median(class_frequencies)
    median_freq_balanced = median / class_frequencies

    return {
        'raw_counts': class_counts,
        'freqs': freqs,
        'inv_freqs': inv_freqs,
        'median_freq_balanced': median_freq_balanced,
    }


###################################################################

def convert_weights_format(pth_file, num_epochs, batch_size, balanced, context_path='resnet18'):
    """
    Converts a PyTorch model weights file to a format compatible with the current training setup.
    This function is a placeholder and should be implemented based on specific requirements.
    Inputs:
        pth_file (str): Path to the input .pth file.
        num_epochs (int): Number of epochs used for training.
        batch_size (int): Batch size used for training.
        balanced (bool): Whether the training was balanced or not.
        context_path (str): The context path used in the model, e.g., 'resnet18'.
    """
    old_model_dict = torch.load(pth_file, map_location='cpu')
    torch.save({
        'model_state_dict': old_model_dict,
        'epoch': num_epochs,
        'batch_size': batch_size,
        'balanced': balanced,
        'context_path': context_path,
    }, pth_file)
    print(f"Converted weights saved to {pth_file} with num_epochs={num_epochs}, batch_size={batch_size}, balanced={balanced}, context_path={context_path}.")

# COMPUTE SAMPLING WEIGHTS

def compute_sampling_weights(dataset, temperature, option='max', num_classes=19, ignore_index=255):
    """
    Compute per-image sampling weights for class-balanced sampling.
    Inputs:
        dataset: dataset object with .labels list
        num_classes: number of semantic classes
        ignore_index: label value to ignore
        temperature: controls how strongly rare classes are favored
                     - 1.0 = original weighting (max bias)
                     - 0.5 = softer bias
                     - 0.0 = uniform sampling (no bias)
        option: controls the weighting strategy
                 - 'max': max rarity of classes present
                 - 'mean': mean rarity of classes present
                 - 'sum': sum rarity of classes present
                 - 'prop': proportional weighting based on pixel-level frequency

    Output:
        torch.DoubleTensor of sampling weights (len(dataset))
    """
    # Initialize class frequency array
    class_freq = np.zeros(num_classes, dtype=np.int64)

    # Check if dataset has labels
    if hasattr(dataset, "labels"):   # GTA5 dataset
        labels_list = dataset.labels
    elif hasattr(dataset, "base_dataset"):   # Augmentation wrapper
        labels_list = dataset.base_dataset.labels
    else:
        raise ValueError("Dataset not supported for class weight computation")

    # Compute global class frequencies
    for label_path in labels_list:
        # Open label from disk, so we need to convert it to the training ID format
        label_img = Image.open(label_path).convert('RGB')
        label = convert_gta5_rgb_to_trainid(label_img)
        unique, counts = np.unique(label, return_counts=True)
        for u, c in zip(unique, counts):
            if u != ignore_index:
                class_freq[u] += c

    # Convert frequencies to inverse weights
    class_weights = 1.0 / (class_freq + 1e-6) # Sum 1e-6 to avoid division by 0 due to underflow
    class_weights = class_weights / class_weights.sum()  # Normalize

    # Apply temperature scaling
    if temperature != 1.0:
        class_weights = np.power(class_weights, temperature) # Element-wise power to the temperature
        class_weights = class_weights / class_weights.sum()  # Renormalize

    # Assign per-image sampling weight
    sample_weights = []
    for label_path in labels_list:
        label = np.array(Image.open(label_path))
        unique = np.unique(label)

        # Weighting strategy
        if option == 'max':
            # Image weight = max rarity of classes present. In this case maximum rarity classes get absolute priority.
            img_weight = max([class_weights[c] for c in unique if c != ignore_index], default=0.0)
        elif option == 'mean':
            # Image weight = mean rarity of classes present. In this case all classes contribute to the image weight. Rarer classes get smoothen out
            img_weight = np.mean([class_weights[c] for c in unique if c != ignore_index])
        elif option == 'sum':
            # Image weight = sum rarity of classes present. Gives even more relevance where multiple rare classes are present.
            img_weight = sum([class_weights[c] for c in unique if c != ignore_index])
        elif option == 'prop':
            # Image weight = proportional weighting. Weights images by the pixel-level frequency of each class in them
            unique, counts = np.unique(label, return_counts=True)
            total_pixels = counts.sum()
            img_weight = sum(
                (counts[i] / total_pixels) * class_weights[c]
                for i, c in enumerate(unique) if c != ignore_index
            )
        else:
            raise ValueError(f"Unknown option '{option}' for sampling weight computation")

        sample_weights.append(img_weight)

    return torch.DoubleTensor(sample_weights)