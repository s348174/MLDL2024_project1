from xml.parsers.expat import model
import torch
import random
from models.deeplabv2.deeplabv2 import ResNetMulti, get_deeplab_v2
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
#from datasets.cityscapes import CityScapesSegmentation #select this for local
from cityscapes import CityScapesSegmentation #select this for colab
#from datasets.gta5 import GTA5 #select this for local
from gta5 import GTA5 #select this for colab
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
from torchvision.transforms import functional as TF
import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.transforms import v2, InterpolationMode
from utils import fast_hist, per_class_iou, compute_class_weights, poly_lr_scheduler, convert_gta5_rgb_to_trainid, compute_gta5_class_weights
import time
from fvcore.nn import FlopCountAnalysis, flop_count_table
import multiprocessing
from torch.amp import autocast, GradScaler

#classes for dataAugmentation

class RandomHorizontalFlipPair:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img, label):
        if random.random() < self.p:
            img = transforms.functional.hflip(img)
            label = transforms.functional.hflip(label)
        return img, label

class RandomGaussianBlur:
    def __init__(self, p=0.5, kernel_size=5, sigma=(0.1, 2.0)):
        self.p = p
        self.blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    def __call__(self, img):
        if random.random() < self.p:
            return self.blur(img)
        return img
    
class RandomMultiply:
    def __init__(self, p=0.5, min_factor=0.7, max_factor=1.3):
        self.p = p
        self.min_factor = min_factor
        self.max_factor = max_factor
    def __call__(self, img):
        if random.random() < self.p:
            factor = random.uniform(self.min_factor, self.max_factor)
            img = transforms.functional.adjust_brightness(img, factor)
        return img

class RandomRotationPair:
    def __init__(self, degrees=10, p=0.5):
        self.degrees = degrees
        self.p = p
    def __call__(self, img, label):
        if random.random() < self.p:
            angle = random.uniform(-self.degrees, self.degrees)
            img = transforms.functional.rotate(img, angle, fill=0)
            label = transforms.functional.rotate(label, angle, fill=255)  # 255 per ignore_index
        return img, label
    
class RandomColorJitter:
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.5):
        self.p = p
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
    def __call__(self, img):
        if random.random() < self.p:
            return self.color_jitter(img)
        return img


# Joint transformation function for image and label for data augmentation

def joint_transform(img, label, do_rotate=False, do_multiply=False, do_blur=False, do_flip=False, do_colorjitter=False):
    img = transforms.Resize((512, 1024))(img)
    label = transforms.Resize((512, 1024), interpolation=Image.NEAREST)(label)
    if do_rotate:
        img, label = RandomRotationPair(p=0.5, degrees=10)(img, label)
    if do_flip:
        img, label = RandomHorizontalFlipPair(p=0.5)(img, label)
    if do_blur:
        img = RandomGaussianBlur(p=0.5, kernel_size=5, sigma=(0.1, 2.0))(img)
    if do_multiply:
        img = RandomMultiply(p=0.5, min_factor=0.7, max_factor=1.3)(img)
    if do_colorjitter:
        img = RandomColorJitter(p=0.5, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)(img)
    # Ensure label is always a 2D tensor of class indices
    label_np = np.array(label)
    if label_np.ndim == 3:
        # If label is RGB or multi-channel, convert to single channel (assume first channel)
        label_np = label_np[..., 0]
    label = Image.fromarray(label_np.astype(np.uint8), mode='L')
    return img, label


# Custom dataset class for augmented segmentation
"""
class AugmentedSegmentationDataset:
    def __init__(self, base_dataset, do_rotate=False, do_multiply=False, do_blur=False, do_flip=False, do_colorjitter=False):
        self.base_dataset = base_dataset
        self.do_rotate = do_rotate
        self.do_multiply = do_multiply
        self.do_blur = do_blur
        self.do_flip = do_flip
        self.do_colorjitter = do_colorjitter

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        if isinstance(label, torch.Tensor):
            label = Image.fromarray(label.numpy().astype(np.uint8), mode='L')
        img, label = joint_transform(
            img, label,
            do_rotate=self.do_rotate,
            do_multiply=self.do_multiply,
            do_blur=self.do_blur,
            do_flip=self.do_flip,
            do_colorjitter=self.do_colorjitter
        )
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        label = torch.from_numpy(np.array(label)).long()
        return img, label

    @property
    def num_classes(self):
        return self.base_dataset.num_classes

    @property
    def classes(self):
        return self.base_dataset.classes
        """

class AugmentedSegmentationDataset(Dataset):
    def __init__(self, base_dataset, do_rotate=False, do_multiply=False, do_blur=False, do_flip=False, do_colorjitter=False):
        self.base_dataset = base_dataset
        self.do_rotate = do_rotate
        self.do_multiply = do_multiply
        self.do_blur = do_blur
        self.do_flip = do_flip
        self.do_colorjitter = do_colorjitter
        self.resize_size = (720, 1280)

        # Prebuild deterministic transform lists
        self.base_transform = transforms.Compose([
            transforms.Resize(self.resize_size),
        ])
        
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]  # img: [3, H, W], label: [H, W] (trainIds)

        # Convert tensors to PIL for augmentation
        img_pil = transforms.ToPILImage()(img)
        label_pil = Image.fromarray(label.cpu().numpy().astype(np.uint8), mode='L')

        # Resize (if not already done in base dataset)
        #img_pil = transforms.Resize(self.resize_size)(img_pil)
        #label_pil = transforms.Resize(self.resize_size, interpolation=Image.NEAREST)(label_pil)

        # Paired augmentations
        if self.do_flip and random.random() < 0.5:
            img_pil = transforms.functional.hflip(img_pil)
            label_pil = transforms.functional.hflip(label_pil)
        if self.do_rotate and random.random() < 0.5:
            angle = random.uniform(-10, 10)
            img_pil = transforms.functional.rotate(img_pil, angle, fill=0)
            label_pil = transforms.functional.rotate(label_pil, angle, fill=255)
        # Single-image augmentations
        if self.do_blur and random.random() < 0.5:
            img_pil = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))(img_pil)
        if self.do_multiply and random.random() < 0.5:
            factor = random.uniform(0.7, 1.3)
            img_pil = transforms.functional.adjust_brightness(img_pil, factor)
        if self.do_colorjitter and random.random() < 0.5:
            img_pil = transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)(img_pil)

        # Convert back to tensor
        img = transforms.ToTensor()(img_pil)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img) # Renormalize after reconverting to torch tensor
        label = torch.from_numpy(np.array(label_pil)).long()

        return img, label

    @property
    def num_classes(self):
        return self.base_dataset.num_classes

    @property
    def classes(self):
        return self.base_dataset.classes
        

#################
# TRAINING DEEPLAB
#################

def deeplab_train(dataset_path, workspace_path, pretrain_imagenet_path, checkpoint=False, balanced=True, num_epochs=50, batch_size=2): 
    # Set the environment variable for PyTorch CUDA memory allocation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    #####################
    # SETUP TRAINING DATA
    #####################

    # Selects the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")
    
    # Paths to the training dataset
    image_dir = dataset_path + "/images/train"
    label_dir = dataset_path + "/gtFine/train"
    # Defining the transforms
    input_transform = transforms.Compose([
        transforms.Resize((512, 1024)),  # Resize to 512x1024 resolution 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Standard normalization for ImageNet ([mean] and [std dev] of RGB channels)
    ])
    target_transform = transforms.Compose([ #prepare the labels
        transforms.Resize((512, 1024), interpolation=Image.NEAREST),  # Resize to 512x1024 resolution
        #transforms.Lambda(lambda img: torch.from_numpy(convert_label_ids_to_train_ids(np.array(img))).long()) 
        transforms.Lambda(lambda img: torch.from_numpy(np.array(img)).long()), # Convert to tensor
    ])
    # Open the dataset
    dataset = CityScapesSegmentation(
        image_dir=image_dir,
        label_dir=label_dir,
        transform=input_transform,
        target_transform=target_transform,   
    )

    #######################
    # DATASET VISUALIZATION
    #######################
    # Visualize the training data
    class_names = dataset.classes
    print(f"Class names: {class_names}")
    print(f"Number of classes:", dataset.num_classes)
    print(f"Number of training samples: {len(dataset.images)}")
    print(f"Number of labels: {len(dataset.labels)}")
    # Display the first image in the dataset
    image = Image.open(dataset.images[0])
    plt.imshow(image)
    plt.title("First image in the dataset")
    plt.axis("off")
    plt.show()

    #####################
    # PREPARING THE MODEL
    #####################
    # Resuming information on eventual checkpoint
    saved_state_dict = torch.load(pretrain_imagenet_path, map_location=device)
    if checkpoint:
        batch_size = saved_state_dict['batch_size']
        if saved_state_dict['balanced']:
            balanced = True
            print("Training with balanced class weights")

    # Define the loader
    max_num_workers = multiprocessing.cpu_count() #colab pro has 4 (the default has just 2) (for Emanuele)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=max_num_workers) 
    print(f"Training with {max_num_workers} workers and batch size {batch_size}.")
        
    # Load the model
    if checkpoint:
        model = get_deeplab_v2(num_classes=len(class_names), pretrain=True, pretrain_model_path=saved_state_dict['model_state_dict']) # The baseline for semantic segmentation
    else:
        model = get_deeplab_v2(num_classes=len(class_names), pretrain=True, pretrain_model_path=pretrain_imagenet_path) # The baseline for semantic segmentation
    model.to(device)  # Move model to device

    # Define loss function
    if balanced: 
        # Evaluate the class weights based on frequencies
        class_weights_dict = compute_class_weights(label_dir, num_classes=dataset.num_classes)
        class_weights = torch.tensor(class_weights_dict['inv_freqs'], dtype=torch.float32).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=255) # Normalized weights for each class
        print("Training with balanced class weights")
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        print("Training without balanced class weights")
    #criterion = torch.nn.BCEWithLogitsLoss()
    #criterion = torch.nn.MSELoss()

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # Optimizer that changes the learning rate at each step 
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler(enabled=True) # It makes the training faster by implementing AMP

    # Resuming checkpoint if available
    if checkpoint:
        optimizer.load_state_dict(saved_state_dict['optimizer_state_dict'])  # Load optimizer state if available
        scaler.load_state_dict(saved_state_dict['scaler'])  # if saved
        current_epoch = saved_state_dict['epoch'] + 1 # Get current epoch from saved state
        print(f"Resuming training from epoch {current_epoch}")
    else:
        current_epoch = 0

    # Move model to device
    print("Moving model to device...")
    model = model.to(device)

    # Polynomial learning rate
    init_lr = 1e-4
    if checkpoint:
        current_iter = saved_state_dict['current_lr_iter']  
    else: 
        current_iter = 0
    max_iter = (num_epochs-current_epoch) * len(train_loader) + current_iter
    print(f"Current iteration: {current_iter}, Max iterations: {max_iter}")

    ###############
    # TRAINING LOOP
    ###############
    for epoch in range(current_epoch, num_epochs):
        model.train()
        for images, labels in train_loader: # For each batch
            images, labels = images.to(device), labels.to(device) # It takes images and labels from the dataloader
            #outputs, _, _ = model(images)
            #loss = criterion(outputs, labels)

            # Mixed precision training with gradient scaling
            optimizer.zero_grad() # Resets the gradient at every batch

            with autocast(device_type="cuda", enabled=True): 
                # When possible (for instance in convolutions but not in losses) it uses float16 instead of float32 
                outputs, _, _ = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward() # It scales dynamically the gradient in order to avoid underflow
            scaler.step(optimizer)
            scaler.update() # Update the weights

            poly_lr_scheduler(optimizer, init_lr, current_iter, max_iter=max_iter)
            current_iter += 1

        # Save model checkpoint
        if epoch % 5 == 0:
            checkpoint_file = workspace_path + "/export/deeplabv2_epoch_{}.pth".format(epoch)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler': scaler.state_dict(),    # If using AMP
                'epoch': epoch,
                'batch_size': batch_size,  # Save the batch size for resuming training
                'balanced': balanced,  # Save whether the model was trained with balanced class weights
                'current_lr_iter': current_iter,  # Save the current iteration for learning rate scheduling
                # 'loss': loss_value,             # Optional
            }, checkpoint_file)
            print(f"DeepLabv2 model saved at epoch {epoch}")
    # Save the model
    export_path = workspace_path + "/export/deeplabv2_final.pth"
    torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': num_epochs,
                'batch_size': batch_size,  # Save the batch size
                'balanced': balanced,  # Save whether the model was trained with balanced class weights
               }, export_path)
    print("Model saved as deeplabv2_final.pth")

#################
# TESTING DEEPLAB
#################

def deeplab_test(dataset_path, model_path, save_dir=None, num_classes=19):
    # Set the environment variable for PyTorch CUDA memory allocation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #################
    # SETUP TEST DATA
    #################
    # Paths to the test dataset
    image_dir = os.path.join(dataset_path, "images/val")
    label_dir = os.path.join(dataset_path, "gtFine/val")
    # Define the transforms
    input_transform = transforms.Compose([
        transforms.Resize((512, 1024)),  # Resize to 512x1024 resolution
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    target_transform = transforms.Compose([
        transforms.Resize((512, 1024), interpolation=Image.NEAREST),  # Resize to 512x1024 resolution
        #transforms.Lambda(lambda img: torch.from_numpy(convert_label_ids_to_train_ids(np.array(img))).long())
        transforms.Lambda(lambda img: torch.from_numpy(np.array(img)).long()), # Convert to tensor
    ])
    # Open the dataset
    test_dataset = CityScapesSegmentation(
        image_dir=image_dir,
        label_dir=label_dir,
        transform=input_transform,
        target_transform=target_transform
    )
    # Create a DataLoader for the test dataset
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) 
    # Batch_size = 1 so that prediction is done 1 sample at a time (necessary to saving segmentation masks) --> individual evaluation

    # Load model info
    saved_state_dict = torch.load(model_path, map_location=device)
    # Load model
    model = get_deeplab_v2(num_classes=num_classes, pretrain=False) # Pretrain is False because weights are inserted in the next line
    model.load_state_dict(saved_state_dict['model_state_dict'])  # Load the model state dict
    model = model.to(device)
    model.eval()

    ########################
    # METRICS INIZIALIZATION
    ########################
    # Initialize correct and total pixel counts
    correct_pixels = 0
    total_pixels = 0
    # Initialize histogram for IoU
    hist = np.zeros((num_classes, num_classes))
    # Initialize lists to store latency and FPS values
    latency = []
    fps = []

    #######################
    # MODEL EVALUATION LOOP
    #######################
    print("Running Deeplab inference...")
    with torch.no_grad():
        for i, (image, label) in enumerate(test_loader):
            image, label = image.to(device), label.to(device)
            # Start timer
            start_time = time.time()

            # Making prediction
            output = model(image)
            pred = torch.argmax(output, dim=1).squeeze(0)  # Get the predicted class for each pixel
            #pred = torch.argmax(output.squeeze(), dim=0)

            # End timer
            end_time = time.time()

            # Flatten predictions and labels (from 2D to 1D)
            pred_flat = pred.view(-1)
            label_flat = label.view(-1)

            # Mask out ignored pixels (ex: 255)
            mask = label_flat != 255
            correct = (pred_flat[mask] == label_flat[mask]).sum().item()
            total = mask.sum().item()
            
            # METRICS CALCULATIONS
            # Update correct and total pixel counts
            correct_pixels += correct
            total_pixels += total
            # Update histogram
            hist += fast_hist(label_flat, pred_flat, num_classes)
            # Calculate latency for this iteration
            latency_i = end_time - start_time
            latency.append(latency_i)
            # Calculate FPS for this iteration
            fps_i = 1 / latency_i
            fps.append(fps_i)

            # Print progress
            if i % 10 == 0:
                print(f"Iteration {i}/{len(test_loader)}, Latency: {latency_i:.4f}s, FPS: {fps_i:.2f}")

    ####################
    # AFTER LOOP METRICS
    ####################
    print(f"\nEvaluation complete on DeepLab with {saved_state_dict['epoch']} epochs, {saved_state_dict['batch_size']} batch size, and balanced={saved_state_dict['balanced']}.")
    # Calculate pixel accuracy
    accuracy = correct_pixels / total_pixels
    print(f"\nPixel Accuracy: {accuracy * 100:.2f}%")

    # Calculate per-class IoU and mean IoU
    iou_per_class = per_class_iou(hist)
    mean_iou = np.nanmean(iou_per_class)
    print(f"Per-class IoU: {iou_per_class}")
    print(f"Mean IoU: {mean_iou}")

    # Calculate mean and standard deviation for latency and FPS
    mean_latency = np.mean(latency) * 1000  # Convert to milliseconds
    std_latency = np.std(latency) * 1000    # Convert to milliseconds
    mean_fps = np.mean(fps)
    std_fps = np.std(fps)
    print(f"Mean Latency: {mean_latency:.2f} ms")
    print(f"Latency Std Dev: {std_latency:.2f} ms")
    print(f"Mean FPS: {mean_fps:.2f}")
    print(f"FPS Std Dev: {std_fps:.2f}")

    # FLOP calculations
    height, width = 512, 1024  # Input image dimensions
    dummy_input = torch.zeros((1, 3, height, width)).to(device)  # Batch size = 1, 3 channels (RGB)
    flops = FlopCountAnalysis(model, dummy_input)
    print(flop_count_table(flops))
    print(f"Total FLOPs: {flops.total() / 1e9:.2f} GFLOPs")

##################
# TRAINING BISENET
##################

from models.bisenet.build_bisenet import BiSeNet

def bisenet_train(dataset_path, workspace_path, pretrained_path, checkpoint=True, balanced=True, num_epochs=50, batch_size=2, context_path='resnet18', augmentation = "00000"):

    #augmentation = "wxyza" w=rotate, x=multiply, y=blur, z=flip, a=color_jitter (1 yes, 0 no)

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #####################
    # SETUP TRAINING DATA
    #####################
    image_dir = os.path.join(dataset_path, "images/train")
    label_dir = os.path.join(dataset_path, "gtFine/train")

    # Crea il dataset base SENZA transform (le augmentation sono gestite dopo dal wrapper)
    base_dataset = CityScapesSegmentation(
        image_dir=image_dir,
        label_dir=label_dir,
        transform=None,
        target_transform=None)
    num_classes = base_dataset.num_classes
    classes_names = base_dataset.classes

    # Augmentation selection
    do_rotate   = augmentation[0] == "1"
    do_multiply = augmentation[1] == "1"
    do_blur     = augmentation[2] == "1"
    do_flip     = augmentation[3] == "1"
    do_colorjitter = augmentation[4] == "1"

    # Wrappa il dataset base con la classe custom
    dataset = AugmentedSegmentationDataset(
        base_dataset,
        do_rotate=do_rotate,
        do_multiply=do_multiply,
        do_blur=do_blur,
        do_flip=do_flip,
        do_colorjitter=do_colorjitter
    )

    #####################
    # PREPARING THE MODEL
    #####################
    # Resuming information on eventual checkpoint
    saved_state_dict = torch.load(pretrained_path, map_location=device)
    if checkpoint:
        batch_size = saved_state_dict['batch_size']
        if saved_state_dict['balanced']:
            balanced = True
            print("Training with balanced class weights")

    # Define the loader
    max_num_workers = multiprocessing.cpu_count() #colab pro has 4 (the default has just 2) (for Emanuele)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=max_num_workers)
    print(f"Training with {max_num_workers} workers and batch size {batch_size}.")

    # Build BiSeNet model with pretrained image
    model = BiSeNet(num_classes=num_classes, context_path=context_path)
    model.to(device)  # Move model to device

    # Define loss function
    if balanced: 
        # Evaluate the class weights based on frequencies
        class_weights_dict = compute_class_weights(label_dir, num_classes=num_classes)
        class_weights = torch.tensor(class_weights_dict['inv_freqs'], dtype=torch.float32).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    # Define optimizer and scaler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler(enabled=True) # AMP

    # Resuming checkpoint if available
    print("BiSeNet pretrain loading...")
    saved_state_dict = torch.load(pretrained_path, map_location=device)
    if checkpoint:
        model.load_state_dict(saved_state_dict['model_state_dict'])  # Load pretrained weights
        optimizer.load_state_dict(saved_state_dict['optimizer_state_dict'])  # Load optimizer state if available
        scaler.load_state_dict(saved_state_dict['scaler'])  # if saved
        current_epoch = saved_state_dict['epoch'] + 1 # Get current epoch from saved state
        print(f"Resuming training from epoch {current_epoch}")
    else:
        current_epoch = 0
        # If the model was trained with a different architecture, we need to adapt the state_dict
        # This is a workaround to load the pretrained weights into the model
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            i_parts = i.split('.')
            if len(i_parts) > 1:  # Validate key format
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
            else:
                print(f"Skipping invalid key format: {i}")
        model.load_state_dict(new_params, strict=False)
        print("Starting training from scratch with pretrained weights")
    
    # Move model to device
    print("Moving model to device...")
    model = model.to(device)

    # Polynomial learning rate decay
    init_lr = 1e-4
    if checkpoint:
        current_iter = saved_state_dict['current_lr_iter']  
    else: 
        current_iter = 0
    max_iter = (num_epochs-current_epoch) * len(train_loader) + current_iter
    print(f"Current iteration: {current_iter}, Max iterations: {max_iter}")

    ###############
    # TRAINING LOOP
    ###############
    for epoch in range(current_epoch, num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast(device_type="cuda", enabled=True):
                outputs = model(images)
                # BiSeNet returns (main, aux1, aux2) in train mode
                if isinstance(outputs, (tuple, list)) and len(outputs) == 3:
                    main_out, aux1, aux2 = outputs
                    loss = criterion(main_out, labels) + 0.4 * criterion(aux1, labels) + 0.4 * criterion(aux2, labels)
                else:
                    loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            poly_lr_scheduler(optimizer, init_lr, current_iter, max_iter=max_iter)
            current_iter += 1
        # Save model checkpoint
        if epoch % 5 == 0:
            checkpoint_file = os.path.join(workspace_path, f"export/bisenet_epoch_{epoch}.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler': scaler.state_dict(),    # If using AMP
                'epoch': epoch,
                'batch_size': batch_size,  # Save the batch size for resuming training
                'balanced': balanced,  # Save whether the model was trained with balanced class weights
                'current_lr_iter': current_iter,  # Save the current iteration for learning rate scheduling
                # 'loss': loss_value,             # Optional
            }, checkpoint_file)
            print(f"BiSeNet model saved at epoch {epoch}")

    # Save final model
    export_path = os.path.join(workspace_path, "export/bisenet_final.pth")
    torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': num_epochs,
                'batch_size': batch_size,  # Save the batch size
                'balanced': balanced,  # Save whether the model was trained with balanced class weights
                'context_path': context_path,  # Save the context path used
               }, export_path)
    print("BiSeNet model saved as bisenet_final.pth")

#################
# TESTING BISENET
#################

def bisenet_test(dataset_path, model_path, num_classes=19, context_path='resnet18'):
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #################
    # SETUP TEST DATA
    #################
    # Paths to the test dataset
    image_dir = os.path.join(dataset_path, "images/val")
    label_dir = os.path.join(dataset_path, "gtFine/val")
    input_transform = transforms.Compose([
        transforms.Resize((512, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    target_transform = transforms.Compose([
        transforms.Resize((512, 1024), interpolation=Image.NEAREST),
        transforms.Lambda(lambda img: torch.from_numpy(np.array(img)).long()),
    ])
    test_dataset = CityScapesSegmentation(
        image_dir=image_dir,
        label_dir=label_dir,
        transform=input_transform,
        target_transform=target_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load model info
    saved_state_dict = torch.load(model_path, map_location=device)
    # Load BiSeNet model
    model = BiSeNet(num_classes=num_classes, context_path=context_path)
    model.load_state_dict(saved_state_dict['model_state_dict'])  # Load the model state dict
    model = model.to(device)
    model.eval()

    ########################
    # METRICS INIZIALIZATION
    ########################
    correct_pixels = 0
    total_pixels = 0
    hist = np.zeros((num_classes, num_classes))
    latency = []
    fps = []

    #######################
    # MODEL EVALUATION LOOP
    #######################
    print("Running BiSeNet inference...")
    with torch.no_grad():
        for i, (image, label) in enumerate(test_loader):
            image, label = image.to(device), label.to(device)
            start_time = time.time()
            output = model(image)
            if isinstance(output, (tuple, list)):
                output = output[0]  # Use main output if model returns aux outputs
            pred = torch.argmax(output, dim=1).squeeze(0)
            end_time = time.time()

            pred_flat = pred.view(-1)
            label_flat = label.view(-1)
            mask = label_flat != 255
            correct = (pred_flat[mask] == label_flat[mask]).sum().item()
            total = mask.sum().item()
            correct_pixels += correct
            total_pixels += total
            hist += fast_hist(label_flat.cpu().numpy(), pred_flat.cpu().numpy(), num_classes)
            latency_i = end_time - start_time
            latency.append(latency_i)
            fps_i = 1 / latency_i
            fps.append(fps_i)
            if i % 10 == 0:
                print(f"Iteration {i}/{len(test_loader)}, Latency: {latency_i:.4f}s, FPS: {fps_i:.2f}")

    ####################
    # AFTER LOOP METRICS
    ####################
    print(f"\nEvaluation complete on BiseNet with {saved_state_dict['epoch']} epochs, {saved_state_dict['batch_size']} batch size, and balanced={saved_state_dict['balanced']}.")
    accuracy = correct_pixels / total_pixels
    print(f"\nPixel Accuracy: {accuracy * 100:.2f}%")
    iou_per_class = per_class_iou(hist)
    mean_iou = np.nanmean(iou_per_class)
    print(f"Per-class IoU: {iou_per_class}")
    print(f"Mean IoU: {mean_iou}")
    mean_latency = np.mean(latency) * 1000
    std_latency = np.std(latency) * 1000
    mean_fps = np.mean(fps)
    std_fps = np.std(fps)
    print(f"Mean Latency: {mean_latency:.2f} ms")
    print(f"Latency Std Dev: {std_latency:.2f} ms")
    print(f"Mean FPS: {mean_fps:.2f}")
    print(f"FPS Std Dev: {std_fps:.2f}")

    # FLOP calculations
    height, width = 512, 1024
    dummy_input = torch.zeros((1, 3, height, width)).to(device)
    flops = FlopCountAnalysis(model, dummy_input)
    print(flop_count_table(flops))
    print(f"Total FLOPs: {flops.total() / 1e9:.2f} GFLOPs")

##########################
# TRAINING BISENET ON GTA5
##########################
def bisenet_on_gta(dataset_path, workspace_path, pretrained_path, checkpoint=False, balanced=True, num_epochs=50, batch_size=2, context_path='resnet18', augmentation = '00000'):
    # Set the environment variable for PyTorch CUDA memory allocation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #####################
    # SETUP TRAINING DATA
    #####################
    image_dir = dataset_path + "/images"
    label_dir = dataset_path + "/labels"

    # Crea il dataset base SENZA transform (le augmentation sono gestite dopo dal wrapper)
    base_dataset = GTA5(
        image_dir=image_dir,
        label_dir=label_dir,
        transform=transforms.Compose([
            transforms.Resize((720, 1280)),  # Resize to 512x1024 resolution
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        target_transform=transforms.Compose([
            transforms.Resize((720, 1280), interpolation=Image.NEAREST),
            transforms.Lambda(lambda img: torch.from_numpy(convert_gta5_rgb_to_trainid(img)).long())
        ])
    )
    
    num_classes = base_dataset.num_classes
    classes_names = base_dataset.classes

    # Augmentation selection
    do_rotate   = augmentation[0] == "1"
    do_multiply = augmentation[1] == "1"
    do_blur     = augmentation[2] == "1"
    do_flip     = augmentation[3] == "1"
    do_colorjitter = augmentation[4] == "1"

    # Wrappa il dataset base con la classe custom
    dataset = AugmentedSegmentationDataset(
        base_dataset,
        do_rotate=do_rotate,
        do_multiply=do_multiply,
        do_blur=do_blur,
        do_flip=do_flip,
        do_colorjitter=do_colorjitter
    )

    #####################
    # PREPARING THE MODEL
    #####################
    # Resuming information on eventual checkpoint
    saved_state_dict = torch.load(pretrained_path, map_location=device)
    if checkpoint:
        batch_size = saved_state_dict['batch_size']
        if saved_state_dict['balanced']:
            balanced = True

    # Define the loader
    max_num_workers = multiprocessing.cpu_count()
    # pin_memory=True is beneficial for GPU training as it speeds up data transfer to CUDA memory.
    # It is not necessary for CPU-only training and can be omitted in such cases.
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=max_num_workers, pin_memory=True)
    print(f"Training with {max_num_workers} workers and batch size {batch_size}.")

    # Build BiSeNet model with pretrained image
    model = BiSeNet(num_classes=dataset.num_classes, context_path=context_path)
    model.to(device)  # Move model to device

    # Define loss function
    if balanced: 
        # Evaluate the class weights based on frequencies
        class_weights_dict = compute_gta5_class_weights(label_dir, num_classes=dataset.num_classes)
        class_weights = torch.tensor(class_weights_dict['inv_freqs'], dtype=torch.float32).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=255) # Normalized weights for each class
        print("Training with balanced class weights")
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        print("Training without balanced class weights")

    # Define optimizer and scaler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler(enabled=True) # AMP

    # Resuming checkpoint if available
    print("BiSeNet pretrain loading...")
    if checkpoint:
        model.load_state_dict(saved_state_dict['model_state_dict'])  # Load pretrained weights
        optimizer.load_state_dict(saved_state_dict['optimizer_state_dict'])  # Load optimizer state if available
        scaler.load_state_dict(saved_state_dict['scaler'])  # if saved
        criterion.load_state_dict(saved_state_dict['criterion_state_dict'])  # Load criterion state
        current_epoch = saved_state_dict['epoch'] + 1  # Get current epoch from saved state
        print(f"Resuming training from epoch {current_epoch}")
    else:
        current_epoch = 0
        # If the model was trained with a different architecture, we need to adapt the state_dict
        # This is a workaround to load the pretrained weights into the model
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            i_parts = i.split('.')
            new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
        model.load_state_dict(new_params, strict=False)
        print("Starting training from scratch with pretrained weights")

    # Move model to device
    print("Moving model to device...")
    model = model.to(device)

    # Polynomial learning rate decay
    init_lr = 1e-4
    if checkpoint:
        current_iter = saved_state_dict['current_lr_iter']  
    else:
        current_iter = 0
    max_iter = (num_epochs-current_epoch) * len(train_loader) + current_iter
    print(f"Current iteration: {current_iter}, Max iterations: {max_iter}")

    ###############
    # TRAINING LOOP
    ###############
    for epoch in range(current_epoch, num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast(device_type="cuda", enabled=True):
                outputs = model(images)
                # BiSeNet returns (main, aux1, aux2) in train mode
                if isinstance(outputs, (tuple, list)) and len(outputs) == 3:
                    main_out, aux1, aux2 = outputs
                    loss = criterion(main_out, labels) + 0.4 * criterion(aux1, labels) + 0.4 * criterion(aux2, labels)
                else:
                    loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # Update polynomial loss scheduler
            poly_lr_scheduler(optimizer, init_lr, current_iter, max_iter=max_iter)
            current_iter += 1

        # Save model checkpoint
        if epoch % 2 == 0:
            checkpoint_file = os.path.join(workspace_path, f"export/bisenet_gta_epoch_{epoch}_{augmentation}.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'criterion_state_dict': criterion.state_dict(),
                'scaler': scaler.state_dict(),    # If using AMP
                'epoch': epoch,
                'batch_size': batch_size,  # Save the batch size for resuming training
                'balanced': balanced,  # Save whether the model was trained with balanced class weights
                'current_lr_iter': current_iter,  # Save the current iteration for learning rate scheduling
                # 'loss': loss_value,             # Optional
            }, checkpoint_file)
            print(f"BiSeNet model on GTA saved at epoch {epoch}")

    # Save final model
    export_path = os.path.join(workspace_path, f"export/bisenet_on_gta_final_{augmentation}.pth")
    torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': num_epochs,
                'batch_size': batch_size,  # Save the batch size
                'balanced': balanced,  # Save whether the model was trained with balanced class weights
                'context_path': context_path,  # Save the context path used
               }, export_path)
    print(f"BiSeNet model saved as bisenet_final_{augmentation}.pth")


from itertools import zip_longest, cycle
class Discriminator(torch.nn.Module):
    def __init__(self, in_channels=19):
        super(Discriminator, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1),
            torch.nn.Upsample(size=(512, 1024), mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        return self.model(x)


def bisenet_adversarial_adaptation(dataset_path, target_path, workspace_path, pretrained_path,
                      checkpoint=False, balanced=True, num_epochs=50, batch_size=2,
                      context_path='resnet18', augmentation='00000', alpha=0.01):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #####################
    # SOURCE DATASET (GTA5)
    #####################
    image_dir = dataset_path + "/images"
    label_dir = dataset_path + "/labels"

    base_dataset = GTA5(
        image_dir=image_dir,
        label_dir=label_dir,
        transform=transforms.Compose([
            transforms.Resize((512, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
        target_transform=transforms.Compose([
            transforms.Resize((512, 1024), interpolation=Image.NEAREST),
            transforms.Lambda(lambda img: torch.from_numpy(convert_gta5_rgb_to_trainid(img)).long())
        ])
    )

    # Augmentation selection
    do_rotate   = augmentation[0] == "1"
    do_multiply = augmentation[1] == "1"
    do_blur     = augmentation[2] == "1"
    do_flip     = augmentation[3] == "1"
    do_colorjitter = augmentation[4] == "1"

    # Wrappa il dataset base con la classe custom
    dataset = AugmentedSegmentationDataset(
        base_dataset,
        do_rotate=do_rotate,
        do_multiply=do_multiply,
        do_blur=do_blur,
        do_flip=do_flip,
        do_colorjitter=do_colorjitter
    )

    #####################
    # TARGET DATASET (Cityscapes)
    #####################
    # Paths to the target dataset
    target_dir = target_path + "/images/train"
    label_tgt_dir = target_path + "/gtFine/train"
    # Defining the transforms
    input_transform = transforms.Compose([
        transforms.Resize((512, 1024)),  # Resize to 512x1024 resolution 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Standard normalization for ImageNet ([mean] and [std dev] of RGB channels)
    ])
    # Open the dataset
    target_dataset = CityScapesSegmentation(
        image_dir=target_dir,
        label_dir=label_tgt_dir,
        transform=input_transform,
    )

    #####################
    # LOADERS
    #####################
    # Resuming information on eventual checkpoint
    saved_state_dict = torch.load(pretrained_path, map_location=device)
    if checkpoint:
        batch_size = saved_state_dict['batch_size']
        if saved_state_dict['balanced']:
            balanced = True

    max_num_workers = multiprocessing.cpu_count()
    train_loader_src = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=max_num_workers, pin_memory=True)
    train_loader_tgt = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, num_workers=max_num_workers, pin_memory=True)
    print(f"Training with {max_num_workers} workers and batch size {batch_size}.")

    #####################
    # MODEL + DISCRIMINATOR
    #####################
    # Define model
    model = BiSeNet(num_classes=dataset.num_classes, context_path=context_path).to(device)

    # Define discriminator
    discriminator = Discriminator(in_channels=19).to(device)

    # Define segmentation loss function, balanced or unbalanced
    if balanced:
        class_weights_dict = compute_gta5_class_weights(label_dir, num_classes=dataset.num_classes)
        class_weights = torch.tensor(class_weights_dict['inv_freqs'], dtype=torch.float32).to(device)
        criterion_seg = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    else:
        criterion_seg = torch.nn.CrossEntropyLoss(ignore_index=255)

    # Define domain loss function
    criterion_domain = torch.nn.BCEWithLogitsLoss()

    # Define optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    # Mixed precision training
    scaler = GradScaler(enabled=True)

    # Resuming checkpoint if available
    print("BiSeNet pretrain loading...")
    if checkpoint:
        model.load_state_dict(saved_state_dict['model_state_dict'])  # Load pretrained weights
        discriminator.load_state_dict(saved_state_dict['discriminator_state_dict'])  # Load discriminator state
        optimizer.load_state_dict(saved_state_dict['optimizer_state_dict'])  # Load optimizer state if available
        optimizer_d.load_state_dict(saved_state_dict['optimizer_d_state_dict'])  # Load optimizer state if available
        scaler.load_state_dict(saved_state_dict['scaler'])  # if saved
        criterion_seg.load_state_dict(saved_state_dict['criterion_seg_state_dict'])  # Load criterion state
        criterion_domain.load_state_dict(saved_state_dict['criterion_domain_state_dict'])  # Load criterion state
        current_epoch = saved_state_dict['epoch'] + 1  # Get current epoch from saved state
        print(f"Resuming training from epoch {current_epoch}")
    else:
        current_epoch = 0
        # If the model was trained with a different architecture, we need to adapt the state_dict
        # This is a workaround to load the pretrained weights into the model
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            i_parts = i.split('.')
            new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
        model.load_state_dict(new_params, strict=False)
        print("Starting training from scratch with pretrained weights")

    # Move model to device
    print("Moving model to device...")
    model = model.to(device)

    # Polynomial learning rate
    init_lr = 1e-4
    if checkpoint:
        current_iter = saved_state_dict['current_lr_iter']  
    else: 
        current_iter = 0
    max_iter = (num_epochs-current_epoch) * len(train_loader_src) + current_iter
    print(f"Current iteration: {current_iter}, Max iterations: {max_iter}")

    print(f"Starting adversarial training with {max_num_workers} workers...")

    #####################
    # TRAINING LOOP
    #####################
    for epoch in range(num_epochs):
        model.train()
        discriminator.train()

        for (src_imgs, src_labels), (tgt_imgs, _) in zip_longest(train_loader_src, cycle(train_loader_tgt)):
            # Check for null values (necessary due to zip_longest)
            if src_imgs is None or src_labels is None:
                break

            src_imgs, src_labels = src_imgs.to(device), src_labels.to(device)
            tgt_imgs = tgt_imgs.to(device)

            optimizer.zero_grad()
            optimizer_d.zero_grad()

            with autocast(device_type="cuda", enabled=True):
                # 1. Segmentation loss (source)
                outputs = model(src_imgs)
                if isinstance(outputs, (tuple, list)) and len(outputs) == 3:
                    main_out, aux1, aux2 = outputs
                    seg_loss = criterion_seg(main_out, src_labels) + \
                               0.4 * criterion_seg(aux1, src_labels) + \
                               0.4 * criterion_seg(aux2, src_labels)
                else:
                    seg_loss = criterion_seg(outputs, src_labels)

                # 2. Adversarial domain loss
                src_pred = main_out.detach()
                tgt_pred = model(tgt_imgs)[0].detach()

                src_d = discriminator(src_pred)
                tgt_d = discriminator(tgt_pred)

                src_domain_loss = criterion_domain(src_d, torch.ones_like(src_d))
                tgt_domain_loss = criterion_domain(tgt_d, torch.zeros_like(tgt_d))
                domain_loss = src_domain_loss + tgt_domain_loss

                # Total loss for generator (segmentation + adversarial)
                outputs_tgt = model(tgt_imgs)[0]
                adv_loss = criterion_domain(discriminator(outputs_tgt), torch.ones_like(tgt_d))

                total_loss = seg_loss + alpha * adv_loss

            # Train segmentation based on loss
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update polynomial loss scheduler
            poly_lr_scheduler(optimizer, init_lr, current_iter, max_iter=max_iter)
            current_iter += 1

            # Train discriminator separately
            scaler.scale(domain_loss).backward()
            scaler.step(optimizer_d)
            scaler.update()

        # Save model checkpoint
        if epoch % 2 == 0:
            checkpoint_file = os.path.join(workspace_path, f"export/bisenet_adversarial_epoch_{epoch}_{augmentation}.pth")
            torch.save({
                'model_state_dict': model.state_dict(), # Save model state
                'discriminator_state_dict': discriminator.state_dict(), # Save discriminator state
                'optimizer_state_dict': optimizer.state_dict(), # Save optimizer state
                'optimizer_d_state_dict': optimizer_d.state_dict(), # Save optimizer on adversarial adaptation state
                'criterion_seg_state_dict': criterion_seg.state_dict(), # Save segmentation criterion state
                'criterion_domain_state_dict': criterion_domain.state_dict(), # Save domain criterion state
                'scaler': scaler.state_dict(),    # If using AMP
                'epoch': epoch,
                'batch_size': batch_size,  # Save the batch size for resuming training
                'balanced': balanced,  # Save whether the model was trained with balanced class weights
                'current_lr_iter': current_iter,  # Save the current iteration for learning rate scheduling
                # 'loss': loss_value,             # Optional
            }, checkpoint_file)
            print(f"BiSeNet model on GTA saved at epoch {epoch}")

    # Save final model
    export_path = os.path.join(workspace_path, f"export/bisenet_adversarial_final_{augmentation}.pth")
    torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': num_epochs,
                'batch_size': batch_size,  # Save the batch size
                'balanced': balanced,  # Save whether the model was trained with balanced class weights
                'context_path': context_path,  # Save the context path used
               }, export_path)
    print(f"BiSeNet model saved as bisenet_adversarial_final_{augmentation}.pth")

    return model