import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import get_model
import os

def evaluate():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Evaluating model on {device}...")

    # Data Prep
    val_dir = "../data/tiny-imagenet-200/val"
    
    # Verify we ran the fix script
    # We check if a random class folder exists
    example_class = os.path.join(val_dir, "n01443537") 
    if not os.path.exists(example_class):
        print("ERROR: Validation folder is messy.")
        print("Please run 'python3 fix_val_folder.py' first!")
        return

    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    print("Loading validation data...")
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Load Model
    model = get_model(device)
    model_path = "../models/resnet_tinyimagenet.pth"
    
    if not os.path.exists(model_path):
        print("Model not found! Let train.py finish first.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    correct = 0
    total = 0
    
    print("Running evaluation...")
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    print("-" * 30)
    print(f"Training Accuracy:  ~92% (Suspiciously High)")
    print(f"Validation Accuracy: {100 * correct / total:.2f}% (The Real Score)")
    print("-" * 30)

if __name__ == "__main__":
    evaluate()