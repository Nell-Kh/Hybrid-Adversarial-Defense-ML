import torch
import torch.nn as nn
import os
from torchvision.utils import save_image
from model import get_model
from dataset import get_dataloaders

# Attack Parameters
EPSILON = 8/255 
ALPHA = 2/255
STEPS = 10

def pgd_attack(model, images, labels, device, eps=EPSILON, alpha=ALPHA, steps=STEPS):
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

def generate_adversarial_samples():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Generating attacks on {device}...")

    # Load Model
    model = get_model(device)
    model_path = "../models/resnet_tinyimagenet.pth"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Warning: Trained model not found. Using random weights.")
    
    model.eval()

    # Directories
    clean_dir = "../data/clean_samples"
    adv_dir = "../data/adversarial_samples"
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(adv_dir, exist_ok=True)

    loader = get_dataloaders()
    count = 0
    MAX_SAMPLES = 50

    print("Starting PGD generation...")
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = pgd_attack(model, images, labels, device)

        clean_preds = model(images).argmax(1)
        adv_preds = model(adv_images).argmax(1)
        
        for i in range(len(images)):
            if count >= MAX_SAMPLES: return

            # Save only successful attacks
            if clean_preds[i] == labels[i] and adv_preds[i] != labels[i]:
                save_image(images[i], os.path.join(clean_dir, f"img_{count}.png"))
                save_image(adv_images[i], os.path.join(adv_dir, f"img_{count}.png"))
                count += 1
                if count % 10 == 0:
                    print(f"Generated {count}/{MAX_SAMPLES} samples")

if __name__ == "__main__":
    generate_adversarial_samples()