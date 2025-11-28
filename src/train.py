import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from tqdm import tqdm
from model import get_model
from dataset import get_dataloaders

# Hyperparameters
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
SAVE_PATH = "../models/resnet_tinyimagenet.pth"

def train():
    # Setup device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load resources
    train_loader = get_dataloaders()
    model = get_model(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Starting training for {NUM_EPOCHS} epochs...")
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=True)
        
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=100.*correct/total)
            
    # Save model
    if not os.path.exists("../models"):
        os.makedirs("../models")
        
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")
    print(f"Total training time: {(time.time() - start_time)/60:.2f} min")

if __name__ == "__main__":
    train()