import torch
import torch.nn as nn
import sys
import os
from tqdm import tqdm

# Adjust path
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import config
from model import get_model
from dataset import get_dataloaders
from attacks.auto_attack import AutoAttackLite

class EnsembleModel(nn.Module):
    def __init__(self, model_a, model_b):
        super(EnsembleModel, self).__init__()
        self.model_a = model_a
        self.model_b = model_b
        
    def forward(self, x):
        # Average probabilities (Soft Voting)
        out_a = self.model_a(x)
        out_b = self.model_b(x)
        return (out_a + out_b) / 2.0

def load_weights(model, path, device):
    print(f"Loading {path}...")
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

def evaluate_ensemble():
    device = config.DEVICE
    print(f"Evaluating Ensemble Defense on {device}...")
    
    # 1. Load Models
    model_a = get_model(device).to(device)
    model_b = get_model(device).to(device)
    
    path_a = os.path.join(config.MODEL_DIR, "resnet_robust.pth")
    path_b = os.path.join(config.MODEL_DIR, "resnet_robust_old.pth")
    
    if not os.path.exists(path_b):
        print("Old model not found! Cannot ensemble.")
        return
        
    load_weights(model_a, path_a, device)
    load_weights(model_b, path_b, device)
    
    # 2. Create Ensemble
    ensemble = EnsembleModel(model_a, model_b)
    ensemble.to(device)
    ensemble.eval()
    
    # 3. Setup Attack
    print("Initializing AutoAttack on Ensemble...")
    attacker = AutoAttackLite(ensemble, device)
    
    # 4. Data
    _, val_loader = get_dataloaders(batch_size=16)
    
    total = 0
    correct_clean = 0
    correct_robust = 0
    MAX_BATCHES = 20
    
    print(f"Running Evaluation (Max {MAX_BATCHES} batches)...")
    
    for i, (images, labels) in enumerate(tqdm(val_loader, total=MAX_BATCHES)):
        if i >= MAX_BATCHES: break
        
        images, labels = images.to(device), labels.to(device)
        total += labels.size(0)
        
        # Clean Acc
        with torch.no_grad():
            outputs = ensemble(images)
            _, predicted = outputs.max(1)
            correct_clean += predicted.eq(labels).sum().item()
            
        # Robust Acc
        adv_images = attacker.attack(images, labels)
        with torch.no_grad():
            adv_outputs = ensemble(adv_images)
            _, adv_predicted = adv_outputs.max(1)
            correct_robust += adv_predicted.eq(labels).sum().item()
            
    print("\n--- ENSEMBLE RESULTS ---")
    print(f"Evaluated {total} images.")
    print(f"Clean Accuracy: {100.*correct_clean/total:.2f}%")
    print(f"Robust Accuracy: {100.*correct_robust/total:.2f}%")

if __name__ == "__main__":
    evaluate_ensemble()
