
import torch
import torch.nn as nn
import argparse
import sys
import os

# Adjust imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
# Add src directory needed for 'import config' inside model.py
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import src.config as config
from src.model import get_model
from src.dataset import get_dataloaders
from src.defenses.mahalanobis import MahalanobisDetector

def simple_pgd(model, images, labels, eps=0.031, alpha=0.007, steps=10):
    images = images.clone().detach().to(config.DEVICE)
    labels = labels.to(config.DEVICE)
    adv_images = images + torch.empty_like(images).uniform_(-eps, eps)
    adv_images = torch.clamp(adv_images, -1, 1).detach()
    loss_fn = nn.CrossEntropyLoss()
    
    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = loss_fn(outputs, labels)
        grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=-1, max=1).detach()
    return adv_images

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=20, help="Number of batches for calibration")
    args = parser.parse_args()

    device = config.DEVICE
    print(f"Calibrating Mahalanobis on {device}...")
    
    # 1. Load Model
    model = get_model(device)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    
    # 2. Load Data
    train_loader, _ = get_dataloaders(batch_size=32)
    
    # 3. Initialize Detector
    detector = MahalanobisDetector(model, device)
    
    if not detector.load_stats():
        print("Stats not found! Running fit first...")
        detector.fit_statistics(train_loader)
        
    if detector.trained:
        print("Detector classifier is already trained. Overwriting...")
        
    # 4. Train Classifier
    # We pass the PGD function so the detector can generate its own training data
    detector.train_classifier(train_loader, simple_pgd, max_batches=args.batches)
    
    print("\nCalibration Complete. The Mahalanobis is now fully operational.")

if __name__ == "__main__":
    main()
