"""
[FILE PURPOSE]
This script trains the Deep Denoising Autoencoder (DAE) to scrub adversarial noise.

[HOW IT WORKS]
1. Loads the Tiny ImageNet dataset.
2. For each batch of clean images, it actively corrupts them using an Adversarial Attack 
   (e.g., FGSM or PGD) to generate realistic adversarial noise.
3. It passes the corrupted image through the Autoencoder.
4. It compares the Autoencoder's output to the ORIGINAL CLEAN IMAGE using MSE Loss.
5. Through backpropagation, the network mathematically learns the mapping required
   to translate Adversarial Space back into pristine Image Space.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from src.dataset import get_dataloaders
from src.model import get_model
from src.attacks.fgsm import FGSMAttacker
from src.defenses.neural_cleaner import DenoisingAutoencoder

# --- HYPERPARAMETERS ---
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-3
EPSILON = 0.03 # The strength of the adversarial noise applied during training

def train_autoencoder():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Training Neural Cleaner on: {device}")
    
    # 1. Load Data
    train_loader, val_loader = get_dataloaders(batch_size=BATCH_SIZE)
    
    # 2. Load the standard Model (We need this to generate the adversarial attacks)
    Standard_model = get_model(device=device, arch='resnet18')
    try:
        Standard_model.load_state_dict(torch.load("models/resnet_tinyimagenet.pth", map_location=device))
        Standard_model.eval() # We are NOT training this model, just using it for attacks
        print("[+] Loaded standard Model for adversarial generation.")
    except Exception as e:
        print("[-] Error: Could not load standard model. Ensure 'models/standard_resnet18.pth' exists.")
        return

    # 3. Initialize the Attacker
    attacker = FGSMAttacker(Standard_model, device, eps=EPSILON)
    
    # 4. Initialize the Autoencoder & Optimizer
    autoencoder = DenoisingAutoencoder().to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss() # Mean Squared Error handles image reconstruction
    
    os.makedirs("models", exist_ok=True)
    best_loss = float('inf')
    
    # --- TRAINING LOOP ---
    for epoch in range(1, EPOCHS + 1):
        autoencoder.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # STEP 1: Generate Corrupted (Adversarial) Images
            # We generate the attack on-the-fly and detach it so gradients don't flow back to the Standard
            corrupted_inputs = attacker.attack(inputs, labels).detach()
            
            # STEP 2: Pass through Autoencoder
            optimizer.zero_grad()
            reconstructed_inputs = autoencoder(corrupted_inputs)
            
            # STEP 3: Calculate Loss against the CLEAN ORIGINAL
            loss = criterion(reconstructed_inputs, inputs)
            
            # STEP 4: Backpropagate
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'MSE Loss': loss.item()})
            
        epoch_loss = running_loss / len(train_loader)
        print(f"[Epoch {epoch}] Average Loss: {epoch_loss:.4f}")
        
        # Save if it's the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_path = "models/neural_cleaner.pth"
            torch.save(autoencoder.state_dict(), save_path)
            print(f"[*] Saved new best model to {save_path} (Loss: {best_loss:.4f})")
            
    print("\n[✔] Neural Cleaner Training Complete!")

if __name__ == "__main__":
    train_autoencoder()
