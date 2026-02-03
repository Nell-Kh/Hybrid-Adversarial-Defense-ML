"""
[FILE PURPOSE]
This is the Main Hub.
Run this file to test your entire arsenal of attacks against the model.
Command: `python3 src/evaluate_suite.py`

[WHAT IT DOES]
1. Loads the protected model (ResNet).
2. Loads a batch of test images.
3. Runs ALL attacks:
   - AutoAttack (The Standard)
   - DeepFool (The Precision Scalpel)
   - Boundary Attack (The Hacker)
4. Prints a report card showing which attacks worked and how messy they were (L2 Norm).
"""

import torch
import torch.nn.functional as F
from model import get_model
from dataset import get_dataloaders
from attacks import BoundaryAttack, DeepFool, AutoAttackLite
import config
from tqdm import tqdm
import numpy as np

def benchmark():
    device = config.DEVICE
    print(f"Benchmarking Attack Suite on {device}...")
    
    # 1. Load Model
    model = get_model(device)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    model.eval()
    
    # 2. Load Data (Small batch for testing)
    _, val_loader = get_dataloaders()
    images, labels = next(iter(val_loader))
    images, labels = images[:8].to(device), labels[:8].to(device) # Only 8 images for speed
    
    # Filter only correctly classified (No point attacking an image that is already wrong)
    with torch.no_grad():
        preds = model(images).argmax(1)
    mask = preds == labels
    if mask.sum() == 0:
        print("No correctly classified images in first batch!")
        return
        
    images = images[mask]
    labels = labels[mask]
    print(f"Attacking {len(images)} images...")
    
    # 3. Define the Arena (The Attacks)
    attacks = {
        "AutoAttack (Standard)": AutoAttackLite(model, device),
        "DeepFool (Min-Norm)": DeepFool(model, device, max_iters=10),
        "Boundary (Black-Box)": BoundaryAttack(model, device, steps=200) # Low steps for quick test
    }
    
    results = {}
    
    # 4. Fight!
    for name, attacker in attacks.items():
        print(f"\nRunning {name}...")
        
        # Run Attack
        try:
            adv_images = attacker.attack(images, labels)
        except Exception as e:
            print(f"Error running {name}: {e}")
            continue
            
        # Metrics
        with torch.no_grad():
            adv_preds = model(adv_images).argmax(1)
        
        # Did we change the label?
        success = (adv_preds != labels).float().mean().item()
        
        # How much did we change the image? (L2 Norm)
        diff = adv_images - images
        l2_norms = diff.view(len(images), -1).norm(dim=1)
        mean_l2 = l2_norms.mean().item()
        
        results[name] = {
            "Success Rate": f"{success:.1%}",
            "Mean L2 Norm": f"{mean_l2:.4f}"
        }
        
    # 5. The Report Card
    print("\n--- BENCHMARK RESULTS ---")
    print(f"{'Attack Name':<25} | {'Success':<10} | {'L2 Norm (Distortion)':<10}")
    print("-" * 55)
    for name, stats in results.items():
        print(f"{name:<25} | {stats['Success Rate']:<10} | {stats['Mean L2 Norm']:<10}")

if __name__ == "__main__":
    benchmark()
