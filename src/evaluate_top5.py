
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os

# Adjust path
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import config
from model import get_model
from dataset import get_dataloaders
from attacks.base import Attacker

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def evaluate_robust_top5(model, device, test_loader):
    model.eval()
    
    # We use a simple PGD attack for evaluation
    # Note: Creating a simple attack here to avoid circular imports if checking quickly
    class SimplePGD(Attacker):
        def __init__(self, model, device, eps=8/255, alpha=2/255, steps=10):
            super().__init__(model, device)
            self.eps = eps
            self.alpha = alpha
            self.steps = steps
            self.loss_fn = nn.CrossEntropyLoss()
            
        def attack(self, images, labels):
            images = images.to(self.device)
            labels = labels.to(self.device)
            adv_images = images.clone().detach()
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, -1, 1).detach()
            
            for _ in range(self.steps):
                adv_images.requires_grad = True
                outputs = self.model(adv_images)
                loss = self.loss_fn(outputs, labels)
                grad = torch.autograd.grad(loss, adv_images)[0]
                adv_images = adv_images.detach() + self.alpha * grad.sign()
                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=-1, max=1).detach()
            return adv_images

    attacker = SimplePGD(model, device)
    
    clean_top1 = 0
    clean_top5 = 0
    adv_top1 = 0
    adv_top5 = 0
    total = 0
    
    print("\n--- Evaluating Top-1 vs Top-5 Robustness ---")
    
    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)
        
        # 1. Clean Accuracy
        with torch.no_grad():
            outputs = model(images)
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            clean_top1 += acc1.item() * images.size(0)
            clean_top5 += acc5.item() * images.size(0)
            
        # 2. Robust Accuracy (Attacked)
        adv_images = attacker.attack(images, labels)
        with torch.no_grad():
            adv_outputs = model(adv_images)
            adv_acc1, adv_acc5 = accuracy(adv_outputs, labels, topk=(1, 5))
            adv_top1 += adv_acc1.item() * images.size(0)
            adv_top5 += adv_acc5.item() * images.size(0)
            
        total += images.size(0)
        
    print(f"\nResults (N={total} images):")
    print(f"Clean Top-1: {clean_top1/total:.2f}%")
    print(f"Clean Top-5: {clean_top5/total:.2f}%")
    print("-" * 30)
    print(f"Robust Top-1: {adv_top1/total:.2f}% (Hard)")
    print(f"Robust Top-5: {adv_top5/total:.2f}% (Fair)")

if __name__ == "__main__":
    device = config.DEVICE
    model = get_model(device)
    
    # Check if robust model exists
    path = os.path.join(os.path.dirname(__file__), '..', 'models', 'resnet_robust.pth')
    if os.path.exists(path):
        print(f"Loading Robust Model from {path}")
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print("Detected checkpoint format, loading 'model_state_dict'...")
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    _, val_loader = get_dataloaders(batch_size=32)
    evaluate_robust_top5(model, device, val_loader)
