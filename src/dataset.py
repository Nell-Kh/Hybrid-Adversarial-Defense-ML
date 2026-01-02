import os
import requests
import zipfile
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import config

def download_and_unzip():
    """Checks if data exists, otherwise downloads and extracts it."""
    zip_path = os.path.join(config.DATA_DIR, "tiny-imagenet-200.zip")
    dataset_path = os.path.join(config.DATA_DIR, "tiny-imagenet-200")

    if os.path.exists(dataset_path) and len(os.listdir(dataset_path)) > 0:
        return

    print("Downloading dataset (this may take a while)...")
    try:
        r = requests.get(config.DATASET_URL, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in tqdm(r.iter_content(chunk_size=1024), unit="KB"):
                if chunk: f.write(chunk)
                
        print("Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(config.DATA_DIR)
            
    except Exception as e:
        print(f"Error preparing data: {e}")
        raise e

def organize_val_folder():
    """
    Restructures the validation folder to be compatible with ImageFolder.
    Moving images from `val/images/` to `val/<class_id>/`.
    """
    val_dir = os.path.join(config.DATA_DIR, "tiny-imagenet-200", "val")
    img_dir = os.path.join(val_dir, "images")
    annot_file = os.path.join(val_dir, "val_annotations.txt")

    if not os.path.exists(img_dir):
        # Already organized
        return

    print("Reorganizing validation folder for PyTorch compatibility...")
    
    with open(annot_file, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split('\t')
        filename = parts[0]
        class_id = parts[1]
        
        class_folder = os.path.join(val_dir, class_id)
        os.makedirs(class_folder, exist_ok=True)
            
        src = os.path.join(img_dir, filename)
        dst = os.path.join(class_folder, filename)
        if os.path.exists(src):
            os.rename(src, dst)
            
    try:
        os.rmdir(img_dir)
    except:
        pass

def get_dataloaders():
    """
    Returns (train_loader, val_loader)
    """
    download_and_unzip()
    organize_val_folder()
    
    # Standard transforms for ResNet
    transform_train = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transform_val = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dir = os.path.join(config.DATA_DIR, 'tiny-imagenet-200', 'train')
    val_dir = os.path.join(config.DATA_DIR, 'tiny-imagenet-200', 'val')
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train data not found at {train_dir}")

    train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform_train)
    val_dataset = torchvision.datasets.ImageFolder(root=val_dir, transform=transform_val)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader