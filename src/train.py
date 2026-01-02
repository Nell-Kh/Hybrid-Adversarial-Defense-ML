import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from tqdm import tqdm
from model import get_model
from dataset import get_dataloaders

# Hyperparameters
NUM_EPOCHS = 15  # Increased slightly
LEARNING_RATE = 0.001
SAVE_PATH = "../models/resnet_tinyimagenet.pth"

def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. FIX: Get Validation Set (Split train into train/val if needed, or use test set)
    # For simplicity, we stick to your loader but add a Scheduler
    train_loader = get_dataloaders()
    model = get_model(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 2. FIX: Learning Rate Scheduler (Drops LR when loss stops decreasing)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    print(f"Starting training for {NUM_EPOCHS} epochs...")
    best_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=True)
        
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=100.*correct/total)
        
        # Step the scheduler
        epoch_avg_loss = running_loss / len(train_loader)
        scheduler.step(epoch_avg_loss)

        # 3. FIX: Save only if it's the best model so far
        if epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss
            if not os.path.exists("../models"): os.makedirs("../models")
            torch.save(model.state_dict(), SAVE_PATH)
            
    print(f"Best model saved with loss: {best_loss:.4f}")

if __name__ == "__main__":
    train()