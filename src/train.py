import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import argparse
from tqdm import tqdm
import config
from model import get_model
from dataset import get_dataloaders

def train(resume=False, dry_run=False):
    print(f"Using device: {config.DEVICE}")
    if dry_run:
        print("--- DRY RUN MODE ACTIVATED ---")
        config.NUM_EPOCHS = 1
    
    # 1. Prepare Data
    print("Loading data...")
    train_loader, val_loader = get_dataloaders()
    print(f"Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches.")

    # 2. Model, Criterion, Optimizer
    model = get_model(config.DEVICE)
    
    start_epoch = 0
    if resume and os.path.exists(config.MODEL_SAVE_PATH):
        print("Resuming from checkpoint...")
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    print(f"Starting training for {config.NUM_EPOCHS} epochs...")
    best_loss = float('inf')
    
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]", leave=True)
        
        for images, labels in loop:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=100.*train_correct/train_total)

            if dry_run:
                print("Dry run: Training batch complete. Breaking.")
                break
        
        avg_train_loss = train_loss / (1 if dry_run else len(train_loader))
        train_acc = 100. * train_correct / train_total

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Val]", leave=False):
                images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                if dry_run:
                    print("Dry run: Validation batch complete. Breaking.")
                    break
        
        avg_val_loss = val_loss / (1 if dry_run else len(val_loader))
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1} Summary: "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Step Scheduler
        scheduler.step(avg_val_loss)

        # Checkpoint
        if avg_val_loss < best_loss:
            print(f"Validation Loss Improved ({best_loss:.4f} -> {avg_val_loss:.4f}). Saving model...")
            best_loss = avg_val_loss
            # In dry run, we skip saving to avoid overwriting good models with garbage, or we save to temp
            if not dry_run:
                torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            else:
                print("Dry run: Skipping model save.")
            
    print(f"Training Complete. Best Validation Loss: {best_loss:.4f}")
    if not dry_run:
        print(f"Model saved to: {config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Run a single batch for testing")
    args = parser.parse_args()
    
    train(resume=args.resume, dry_run=args.dry_run)