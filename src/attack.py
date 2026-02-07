import torch
import torch.nn as nn
import os
from torchvision.utils import save_image
from tqdm import tqdm
import config
from model import get_model
from dataset import get_dataloaders

def pgd_attack(model, images, labels, device, eps=config.ATTACK_EPSILON, alpha=config.ATTACK_ALPHA, steps=config.ATTACK_STEPS):
    """
    Implementation of Projected Gradient Descent (PGD) attack.
    """
    adv_images = images.clone().detach().to(device)
    labels = labels.to(device)

    # Random start
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
    adv_images = torch.clamp(adv_images, 0, 1)

    loss_fn = nn.CrossEntropyLoss()

    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = loss_fn(outputs, labels)

        grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + alpha * grad.sign()
        
        # Project back to epsilon ball
        delta = torch.clamp(adv_images - images.to(device), min=-eps, max=eps)
        adv_images = torch.clamp(images.to(device) + delta, min=0, max=1).detach()

    return adv_images

def generate_adversarial_samples(n_samples=50):
    print(f"Generating attacks on {config.DEVICE}...")

    # Load Model
    model = get_model(config.DEVICE)
    if os.path.exists(config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    else:
        print("Warning: Trained model not found. Using random weights (Attacks will be meaningless).")
    
    model.eval()

    # Directories
    os.makedirs(config.CLEAN_SAMPLES_DIR, exist_ok=True)
    os.makedirs(config.ADV_SAMPLES_DIR, exist_ok=True)

    # Load Data (Validation set is better for generating examples to test on)
    _, val_loader = get_dataloaders()
    
    count = 0
    print(f"Starting PGD generation (Target: {n_samples} samples)...")
    
    for images, labels in val_loader:
        if count >= n_samples: break
        
        images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
        
        # Filter for correctly classified images first
        with torch.no_grad():
            clean_outputs = model(images)
            clean_preds = clean_outputs.argmax(1)
            
        correct_mask = (clean_preds == labels)
        if not correct_mask.any(): continue
        
        # Only attack correctly classified images
        images = images[correct_mask]
        labels = labels[correct_mask]
        
        adv_images = pgd_attack(model, images, labels, config.DEVICE)
        adv_preds = model(adv_images).argmax(1)
        
        for i in range(len(images)):
            if count >= n_samples: break

            # Save if attack was successful (or we just want samples regardless?)
            # Usually we want successful attacks for highlighting
            if adv_preds[i] != labels[i]:
                save_image(images[i], os.path.join(config.CLEAN_SAMPLES_DIR, f"img_{count}.png"))
                save_image(adv_images[i], os.path.join(config.ADV_SAMPLES_DIR, f"img_{count}.png"))
                count += 1
                if count % 10 == 0:
                    print(f"Generated {count}/{n_samples} samples")

if __name__ == "__main__":
    generate_adversarial_samples()