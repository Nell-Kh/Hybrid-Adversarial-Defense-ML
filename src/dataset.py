import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import requests
import zipfile
from tqdm import tqdm

DATA_DIR = "../data"
URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
BATCH_SIZE = 64

def download_and_unzip():
    """Checks if data exists, otherwise downloads and extracts it."""
    zip_path = os.path.join(DATA_DIR, "tiny-imagenet-200.zip")
    dataset_path = os.path.join(DATA_DIR, "tiny-imagenet-200")

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    if os.path.exists(dataset_path) and len(os.listdir(dataset_path)) > 0:
        return

    print("Downloading dataset...")
    try:
        r = requests.get(URL, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in tqdm(r.iter_content(chunk_size=1024)):
                if chunk: f.write(chunk)
                
        print("Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
            
    except Exception as e:
        print(f"Error preparing data: {e}")

def get_dataloaders():
    download_and_unzip()
    
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dir = os.path.join(DATA_DIR, 'tiny-imagenet-200', 'train')
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Data not found at {train_dir}")

    train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
    
    return DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)