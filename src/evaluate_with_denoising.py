import torch
import torch.nn as nn
import sys
import os
import argparse
from tqdm import tqdm

# Adjust path
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import config
from model import get_model
from dataset import get_dataloaders
from attacks.auto_attack import AutoAttackLite
from defenses.feature_denoising import DenoisingModel

def evaluate_defense():
    device = config.DEVICE
    print(f"Evaluating Feature Denoising Defense on {device}...")
    
    # 1. Load standard Model
    base_model = get_model(device)
    path = os.path.join(config.MODEL_DIR, "resnet_robust.pth")
    
    if not os.path.exists(path):
        print("Model not found!")
        return

    print(f"Loading Base Model: {path}")
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        base_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        base_model.load_state_dict(checkpoint)
        
    # 2. Wrap with Denoiser
    print("Applying Median Filter Denoising...")
    model = DenoisingModel(base_model, denoiser_type='median')
    model.to(device)
    model.eval()
    
    # 3. Setup Attack
    print("Initializing AutoAttack (The 'Teacher' of the test)...")
    attacker = AutoAttackLite(model, device) # Note: We attack the *defended* model
    
    # 4. Data
    _, val_loader = get_dataloaders(batch_size=16) # Smaller batch for unfolding overhead
    
    total = 0
    correct_clean = 0
    correct_robust = 0
    
    MAX_BATCHES = 20 # Limit for speed (approx 300 images)
    
    print(f"Running Evaluation (Max {MAX_BATCHES} batches)...")
    
    for i, (images, labels) in enumerate(tqdm(val_loader)):
        if i >= MAX_BATCHES: break
        
        images, labels = images.to(device), labels.to(device)
        total += labels.size(0)
        
        # A. Clean Accuracy (Did blurring hurt us?)
        with torch.no_grad():
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct_clean += predicted.eq(labels).sum().item()
            
        # B. Robust Accuracy (Did blurring help us?)
        # Filter correctly classified only for attribution? 
        # Be strict: Standard Accuracy (Attack everything) -> Robust Acc
        
        adv_images = attacker.attack(images, labels)
        
        with torch.no_grad():
            adv_outputs = model(adv_images)
            _, adv_predicted = adv_outputs.max(1)
            correct_robust += adv_predicted.eq(labels).sum().item()
            
    print("\n--- RESULTS ---")
    print(f"Evaluated {total} images.")
    print(f"Clean Accuracy (with Denoising): {100.*correct_clean/total:.2f}%")
    print(f"Robust Accuracy (with Denoising): {100.*correct_robust/total:.2f}%")
    print("----------------")
    
    if (100.*correct_robust/total) > 61.5:
        print("✅ SUCCESS: Denoising improved robustness!")
    else:
        print("⚠️ NOTE: Improvement might require retraining with the denoiser.")

if __name__ == "__main__":
    evaluate_defense()
