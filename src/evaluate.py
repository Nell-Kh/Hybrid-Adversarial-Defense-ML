import torch
import config
from model import get_model
from dataset import get_dataloaders
from tqdm import tqdm
import os

def evaluate():
    print(f"Evaluating model on {config.DEVICE}...")

    # Load Model
    model = get_model(config.DEVICE)
    
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print(f"Model not found at {config.MODEL_SAVE_PATH}! Please run train.py first.")
        return
        
    print(f"Loading weights from {config.MODEL_SAVE_PATH}...")
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    model.eval()

    # Get Validation Data
    print("Loading data...")
    # we discard the train loader
    _, val_loader = get_dataloaders()
    
    correct = 0
    total = 0
    
    print("Running evaluation...")
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    acc = 100 * correct / total
    print("-" * 30)
    print(f"Validation Accuracy: {acc:.2f}%")
    print("-" * 30)
    
    # Gate Check
    if acc < 30.0:
        print("❌ FAIL: Model is not smart enough (>30% required). Retrain needed.")
    else:
        print("✅ PASS: Model quality is acceptable for research.")

if __name__ == "__main__":
    evaluate()