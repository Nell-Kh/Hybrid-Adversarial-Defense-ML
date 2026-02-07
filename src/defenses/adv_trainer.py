
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import argparse
from tqdm import tqdm

# Adjust imports based on your project structure
# Assuming we are in src/defenses/ or running from root as python src/defenses/adv_trainer.py
import sys
# Add project root needed for 'src.config'
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
# Add src directory needed for 'import config' inside model.py
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import src.config as config
from src.model import get_model
from src.dataset import get_dataloaders

def pgd_attack(model, images, labels, eps=0.031, alpha=0.007, steps=7):
    """
    Fast PGD for Adversarial Training (Madry et al.)
    # CONCEPT: Efficiency Trade-off
    # A full attack (to break the model) needs 40+ steps.
    # But if we did that during training, it would take weeks.
    # We use a "Weakened Virus" (7 steps) to train the immune system faster.
    """
    images = images.clone().detach().to(config.DEVICE)
    labels = labels.to(config.DEVICE)
    
    # 1. Random Start (Important for training diversity)
    adv_images = images + torch.empty_like(images).uniform_(-eps, eps)
    adv_images = torch.clamp(adv_images, -1, 1).detach()
    
    loss_fn = nn.CrossEntropyLoss()
    
    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = loss_fn(outputs, labels)
        
        grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
        
        adv_images = adv_images.detach() + alpha * grad.sign()
        
        # Clip to epsilon ball
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        # Clip to valid image range [-1, 1]
        adv_images = torch.clamp(images + delta, min=-1, max=1).detach()
        
    return adv_images

def train_robust_model(epochs=5, batch_size=64):
    device = config.DEVICE
    print(f"Starting Adversarial Training on {device}...")
    print(f"Target: {epochs} epochs")
    
    # 1. Load Data
    train_loader, val_loader = get_dataloaders(batch_size=batch_size)
    
    # 2. Load Model (Start from the standard pre-trained one to save time)
    model = get_model(device)
    if os.path.exists(config.MODEL_SAVE_PATH):
        print("Loading pre-trained standard model as baseline...")
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Warning: No pre-trained model found. Starting from scratch.")
        
    model.train()
    
    # 3. Setup Optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    # Scheduler: Drop learning rate by 10x every 10 epochs (Standard practice for ResNet)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    # 4. Training Loop
    save_path = os.path.join(config.MODEL_DIR, "resnet_robust.pth")
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        # Progress bar
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            # --- THE MAGIC STEP (The Vaccine) ---
            # Normal training: Input -> Model -> Loss
            # Adversarial Training: Input -> Attack Generator -> Adversarial Input -> Model -> Loss
            
            model.eval() # 1. Freeze model to generate the attack (don't update weights yet)
            adv_images = pgd_attack(model, images, labels)
            model.train() # 2. Unfreeze model to learn how to resist the attack
            # ----------------------
            
            optimizer.zero_grad()
            outputs = model(adv_images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            loop.set_postfix(acc=100.*correct/total, loss=total_loss/total, lr=optimizer.param_groups[0]['lr'])
        
        # Step the scheduler at the end of epoch
        scheduler.step()
            
        print(f"Epoch {epoch+1} Complete. Robust Acc: {100.*correct/total:.2f}% | Time: {time.time()-start_time:.0f}s")
        
        # Save every epoch
        torch.save(model.state_dict(), save_path)
        print(f"Saved robust model to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    args = parser.parse_args()
    
    train_robust_model(epochs=args.epochs, batch_size=args.batch_size)
