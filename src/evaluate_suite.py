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

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Adversarial Attack Benchmark Suite")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of images to attack")
    parser.add_argument("--attacks", type=str, default="all", help="Comma-separated list of attacks (auto,deepfool,boundary) or 'all'")
    parser.add_argument("--max-iters", type=int, default=50, help="Max iterations for DeepFool")
    parser.add_argument("--steps", type=int, default=200, help="Steps for Boundary Attack")
    parser.add_argument("--model", type=str, default="default", help="Path to model (.pth) or 'default'")
    return parser.parse_args()

def benchmark():
    args = parse_args()
    device = config.DEVICE
    print(f"Benchmarking Attack Suite on {device}...")
    
    # 1. Load Model
    model = get_model(device)
    if args.model == "default":
        model_path = config.MODEL_SAVE_PATH
    else:
        model_path = args.model
        
    print(f"Loading model from: {model_path}")
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print("Detected checkpoint format, loading 'model_state_dict'...")
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # 2. Load Data
    _, val_loader = get_dataloaders()
    images, labels = next(iter(val_loader))
    images, labels = images[:args.batch_size].to(device), labels[:args.batch_size].to(device)
    
    # Filter only correctly classified 
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
    all_attacks = {
        "auto": ("AutoAttack (Standard)", AutoAttackLite(model, device)),
        "deepfool": ("DeepFool (Min-Norm)", DeepFool(model, device, max_iters=args.max_iters)),
        "boundary": ("Boundary (Black-Box)", BoundaryAttack(model, device, steps=args.steps))
    }
    
    if args.attacks == "all":
        target_attacks = all_attacks
    else:
        requested = args.attacks.split(",")
        target_attacks = {k: v for k, v in all_attacks.items() if k in requested}
    
    results = {}
    
    # 4. Fight!
    for key, (name, attacker) in target_attacks.items():
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
