
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import time
import argparse
from tqdm import tqdm
import sys

# Adjust imports based on your project structure
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import config
from model import get_model
from dataset import get_dataloaders

def trades_loss(model, x_natural, y, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0, distance='l_inf'):
    """
    TRADES Loss function (Zhang et al., 2019)
    Defines the tradeoff between clean accuracy and robust accuracy.
    """
    # Define KL-divergence loss
    criterion_kl = nn.KLDivLoss(reduction='sum')
    model.eval()
    
    batch_size = len(x_natural)
    
    # Generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(config.DEVICE).detach()
    
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, -1.0, 1.0)
    else:
        # L2 distance not strictly required for this project but good to have
        pass
    
    model.train()
    
    x_adv = x_adv.detach().requires_grad_(False)
    
    # Calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    
    loss = loss_natural + beta * loss_robust
    return loss, loss_natural, loss_robust

def train_advanced(epochs=50, batch_size=64, arch='resnet18', beta=6.0, dry_run=False):
    device = config.DEVICE
    print(f"Starting ADVANCED Training (TRADES) on {device}...")
    print(f"Arch: {arch} | Epochs: {epochs} | Beta: {beta}")
    print("NOTE: 'Train Clean Acc' is NOT robust accuracy. It is how well the model learns the training data.")
    
    
    # 1. Load Data
    train_loader, val_loader = get_dataloaders(batch_size=batch_size)
    if dry_run:
        print("DRY RUN: truncating data")
        train_loader = [next(iter(train_loader))]
        val_loader = [next(iter(val_loader))]
    
    # 2. Load Model
    model = get_model(device, arch=arch)
    model.train()
    
    # 3. Optimizer & Scheduler
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    # Cosine Annealing is often better for long training runs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    
    # 4. AMP Scaler for GPU acceleration
    use_amp = torch.cuda.is_available() and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    save_path = os.path.join(config.MODEL_DIR, f"{arch}_trades_beta{beta}.pth")
    best_acc = 0
    
    for epoch in range(epochs):
        total_loss = 0
        total_natural_loss = 0
        total_robust_loss = 0
        correct = 0
        total = 0
        
        start_time = time.time()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Use AMP for speed if available
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, loss_nat, loss_rob = trades_loss(model, images, labels, optimizer, beta=beta)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Metrics
            total_loss += loss.item()
            total_natural_loss += loss_nat.item()
            total_robust_loss += loss_rob.item()
            
            with torch.no_grad():
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            loop.set_postfix({'Train Clean Acc': 100.*correct/total, 'loss': loss.item(), 'rob_loss': loss_rob.item()})
            
            
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Train Clean Acc: {100.*correct/total:.2f}% | Time: {time.time()-start_time:.0f}s")
        
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'acc': correct/total,
        }, save_path)
        print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--arch", type=str, default='resnet18', choices=['resnet18', 'resnet50', 'wide_resnet50_2'])
    parser.add_argument("--beta", type=float, default=6.0, help="TRADES regularization parameter (higher = more robust, less clean acc)")
    parser.add_argument("--dry-run", action="store_true", help="Run a single batch to verify code works")
    
    args = parser.parse_args()
    
    train_advanced(epochs=args.epochs, batch_size=args.batch_size, arch=args.arch, beta=args.beta, dry_run=args.dry_run)
