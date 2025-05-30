import torch
from models.deeplabv2.deeplabv2 import ResNetMulti, get_deeplab_v2
from torch.utils.data import DataLoader
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
from utils import fast_hist, per_class_iou, compute_class_weights, poly_lr_scheduler, convert_gta5_rgb_to_trainid, compute_gta5_class_weights
import time
from fvcore.nn import FlopCountAnalysis, flop_count_table
import multiprocessing
from torch.amp import autocast, GradScaler

#################
# TRAINIG DEEPLAB
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
    if checkpoint: batch_size = saved_state_dict['batch_size']
    max_num_workers = multiprocessing.cpu_count() #colab pro has 4 (the default has just 2) (for Emanuele)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=max_num_workers) 
    print(f"Using {batch_size} as batch size.")
    print(f"Using {max_num_workers} workers for data loading.")
        
    # Load the model
    if checkpoint:
        model = get_deeplab_v2(num_classes=len(class_names), pretrain=True, pretrain_model_path=saved_state_dict['model_state_dict']) # The baseline for semantic segmentation
    else:
        model = get_deeplab_v2(num_classes=len(class_names), pretrain=True, pretrain_model_path=pretrain_imagenet_path) # The baseline for semantic segmentation
    
    # Define loss function
    if balanced: 
        # Evaluate the class weights based on frequencies
        class_weights_dict = compute_class_weights(label_dir, num_classes=dataset.num_classes)
        class_weights = torch.tensor(class_weights_dict['inv_freqs'], dtype=torch.float32).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=255) # Normalized weights for each class
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
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
        current_epoch = saved_state_dict['epoch']  # Get current epoch from saved state
        print(f"Resuming training from epoch {current_epoch}")
    else:
        current_epoch = 0

    # Move model to device
    print("Moving model to device...")
    model = model.to(device)

    # Polynomial learning rate
    init_lr = 1e-4
    max_iter = (num_epochs-current_epoch) * len(train_loader)
    current_iter = saved_state_dict['current_lr_iter'] if checkpoint else 0

    ###############
    # TRAINING LOOP
    ###############
    for epoch in range(current_epoch, num_epochs-current_epoch):
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
                'bacth_size': batch_size,  # Save the batch size for resuming training
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

def bisenet_train(dataset_path, workspace_path, pretrained_path, checkpoint=False, balanced=True, num_epochs=50, batch_size=2, context_path='resnet18'):
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #####################
    # SETUP TRAINING DATA
    #####################
    image_dir = os.path.join(dataset_path, "images/train")
    label_dir = os.path.join(dataset_path, "gtFine/train")
    input_transform = transforms.Compose([
        transforms.Resize((512, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    target_transform = transforms.Compose([
        transforms.Resize((512, 1024), interpolation=Image.NEAREST),
        transforms.Lambda(lambda img: torch.from_numpy(np.array(img)).long()),
    ])
    dataset = CityScapesSegmentation(
        image_dir=image_dir,
        label_dir=label_dir,
        transform=input_transform,
        target_transform=target_transform,
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
    if checkpoint: batch_size = saved_state_dict['batch_size']
    max_num_workers = multiprocessing.cpu_count() #colab pro has 4 (the default has just 2) (for Emanuele)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=max_num_workers)

    # Build BiSeNet model with pretrained image
    model = BiSeNet(num_classes=dataset.num_classes, context_path=context_path)

    # Define loss function
    if balanced: 
        # Evaluate the class weights based on frequencies
        class_weights_dict = compute_class_weights(label_dir, num_classes=dataset.num_classes)
        class_weights = torch.tensor(class_weights_dict['inv_freqs'], dtype=torch.float32).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=255) # Normalized weights for each class
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
        current_epoch = saved_state_dict['epoch']  # Get current epoch from saved state
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
    max_iter = (num_epochs-current_epoch) * len(train_loader)
    current_iter = saved_state_dict['current_lr_iter'] if checkpoint else 0

    ###############
    # TRAINING LOOP
    ###############
    for epoch in range(current_epoch, num_epochs - current_epoch):
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
                'bacth_size': batch_size,  # Save the batch size for resuming training
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
def bisenet_on_gta(dataset_path, workspace_path, pretrained_path, checkpoint=False, balanced=True, num_epochs=50, batch_size=2, context_path='resnet18'):
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #####################
    # SETUP TRAINING DATA
    #####################
    image_dir = dataset_path + "/images"
    label_dir = dataset_path + "/labels"
    input_transform = transforms.Compose([
        transforms.Resize((720, 1280)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    target_transform = transforms.Compose([
        transforms.Resize((720, 1280), interpolation=Image.NEAREST),
        transforms.Lambda(lambda img: torch.from_numpy(convert_gta5_rgb_to_trainid(img)).long()),
    ])
    dataset = GTA5(
        image_dir=image_dir,
        label_dir=label_dir,
        transform=input_transform,
        target_transform=target_transform,
    )

    # --- Add this block here to check pixels labels in sample images ---
    N = 10
    for i in range(N):
        label_path = dataset.labels[i]
        label_img = Image.open(label_path).convert("RGB")
        label_np = convert_gta5_rgb_to_trainid(label_img)
        unique, counts = np.unique(label_np, return_counts=True)
        print(f"GTA5 Sample {i}: {dict(zip(unique, counts))}")
    # ---------------------------

    """
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
    # Display the first label in the dataset
    label = Image.open(dataset.labels[0])
    plt.imshow(label)
    plt.title("First label in the dataset")
    plt.axis("off")
    plt.show()
    """

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
    if checkpoint: batch_size = saved_state_dict['batch_size']
    max_num_workers = multiprocessing.cpu_count() #colab pro has 4 (the default has just 2) (for Emanuele)
    # pin_memory=True is beneficial for GPU training as it speeds up data transfer to CUDA memory.
    # It is not necessary for CPU-only training and can be omitted in such cases.
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=max_num_workers, pin_memory=True)

    # Build BiSeNet model with pretrained image
    model = BiSeNet(num_classes=dataset.num_classes, context_path=context_path)

    # Define loss function
    if balanced: 
        # Evaluate the class weights based on frequencies
        class_weights_dict = compute_class_weights(label_dir, num_classes=dataset.num_classes)
        class_weights = torch.tensor(class_weights_dict['inv_freqs'], dtype=torch.float32).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=255) # Normalized weights for each class
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
        current_epoch = saved_state_dict['epoch']  # Get current epoch from saved state
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
    max_iter = (num_epochs-current_epoch) * len(train_loader)
    current_iter = saved_state_dict['current_lr_iter'] if checkpoint else 0

    ###############
    # TRAINING LOOP
    ###############
    for epoch in range(current_epoch, num_epochs - current_epoch):
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
            checkpoint_file = os.path.join(workspace_path, f"export/bisenet_gta_epoch_{epoch}.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler': scaler.state_dict(),    # If using AMP
                'epoch': epoch,
                'bacth_size': batch_size,  # Save the batch size for resuming training
                'balanced': balanced,  # Save whether the model was trained with balanced class weights
                'current_lr_iter': current_iter,  # Save the current iteration for learning rate scheduling
                # 'loss': loss_value,             # Optional
            }, checkpoint_file)
            print(f"BiSeNet model on GTA saved at epoch {epoch}")

    # Save final model
    export_path = os.path.join(workspace_path, "export/bisenet_on_gta_final.pth")
    torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': num_epochs,
                'batch_size': batch_size,  # Save the batch size
                'balanced': balanced,  # Save whether the model was trained with balanced class weights
                'context_path': context_path,  # Save the context path used
               }, export_path)
    print("BiSeNet model saved as bisenet_final.pth")
