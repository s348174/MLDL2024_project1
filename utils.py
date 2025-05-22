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
