
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
import torch.nn.functional as F

def trades_loss(model, x_natural, y, optimizer, step_size=0.007, epsilon=0.031, perturb_steps=7, beta=6.0):
    """
    TRADES: TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization.
    Mathematically superior to standard PGD. It balances clean accuracy (CrossEntropy)
    with adversarial robustness (KL-Divergence between clean and adv logits).
    """
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss(reduction='sum')
    
    model.eval()
    batch_size = len(x_natural)
    
    # 1. Generate adversarial example (Maximize KL-Divergence instead of Error)
    # Start with small random noise
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(config.DEVICE).detach()
    
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            clean_softmax = F.softmax(model(x_natural), dim=1)
            adv_log_softmax = F.log_softmax(model(x_adv), dim=1)
            loss_kl = criterion_kl(adv_log_softmax, clean_softmax)
            
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, -1.0, 1.0)
        
    model.train()
    x_adv = x_adv.detach()
    
    # 2. Calculate TRADES objective
    optimizer.zero_grad()
    logits = model(x_natural)
    
    # Standard Loss (Accuracy on Clean Images: "Don't forget what a dog is")
    loss_natural = criterion_ce(logits, y)
    
    # Robust Loss (Consistency on Adv Images: "Don't change your mind when noise is added")
    clean_softmax = F.softmax(logits, dim=1) # Reuse logits
    adv_log_softmax = F.log_softmax(model(x_adv), dim=1)
    loss_robust = (1.0 / batch_size) * criterion_kl(adv_log_softmax, clean_softmax)
    
    # Final Balanced Objective
    loss = loss_natural + beta * loss_robust
    return loss, logits

def train_robust_model(epochs=5, batch_size=64):
    device = config.DEVICE
    print(f"Starting Adversarial Training on {device}...")
    print(f"Target: {epochs} epochs")
    
    # 1. Load Data
    train_loader, val_loader = get_dataloaders(batch_size=batch_size)
    
    # 2. Load Model (Start from the standard pre-trained one to save time)
    model = get_model(device)
    if os.path.exists(config.MODEL_SAVE_PATH):
        print("Loading pre-trained standard model as standard...")
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
            
            # --- THE MAGIC STEP (TRADES Vaccine) ---
            loss, logits = trades_loss(model, images, labels, optimizer)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
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
