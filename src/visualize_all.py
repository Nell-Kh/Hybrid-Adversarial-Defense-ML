import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import config
from model import get_model
from dataset import get_dataloaders
from attacks import AutoAttackLite, DeepFool, BoundaryAttack

# Ensure outputs directory exists
os.makedirs(os.path.join(config.DATA_DIR, "viz_results"), exist_ok=True)

def denorm(x):
    """Convert [-1, 1] tensor to [0, 1] for plotting."""
    return (x * 0.5 + 0.5).clamp(0, 1)

def visualize_all():
    device = config.DEVICE
    print(f"Generating Attack Gallery on {device}...")
    
    # 1. Setup
    model = get_model(device)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    model.eval()
    
    # Get ONE good image
    _, val_loader = get_dataloaders()
    images, labels = next(iter(val_loader))
    images, labels = images.to(device), labels.to(device)
    
    # Find an image that is correctly classified initially
    with torch.no_grad():
        preds = model(images).argmax(1)
    
    target_idx = -1
    for i in range(len(images)):
        if preds[i] == labels[i]:
            target_idx = i
            break
            
    if target_idx == -1:
        print("No correctly classified images found in batch.")
        return

    img = images[target_idx:target_idx+1]
    label = labels[target_idx:target_idx+1]
    original_class = label.item()
    
    print(f"Selected Image Index: {target_idx}, Class: {original_class}")

    # 2. Run Attacks
    attacks = {
        "AutoAttack": AutoAttackLite(model, device),
        "DeepFool": DeepFool(model, device, max_iters=50),
        "Boundary": BoundaryAttack(model, device, steps=200)
    }
    
    results = {}
    
    for name, attacker in attacks.items():
        print(f"Running {name}...")
        adv_img = attacker.attack(img, label)
        
        with torch.no_grad():
            pred_class = model(adv_img).argmax(1).item()
            
        results[name] = {
            "image": adv_img,
            "pred": pred_class,
            "success": pred_class != original_class
        }

    # 3. Plotting
    print("Plotting results...")
    rows = len(attacks) + 1
    cols = 3 # Image, Noise, Details
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4*rows))
    
    # Row 0: Original
    ax_img = axes[0, 0]
    ax_noise = axes[0, 1]
    ax_text = axes[0, 2]
    
    # Plot Original
    ax_img.imshow(denorm(img).cpu().squeeze().permute(1, 2, 0))
    ax_img.set_title(f"Original\nClass: {original_class}")
    ax_img.axis('off')
    
    # Plot Noise (None)
    ax_noise.imshow(torch.zeros_like(img).cpu().squeeze().permute(1, 2, 0))
    ax_noise.set_title("No Noise")
    ax_noise.axis('off')
    
    # Text info
    ax_text.text(0.1, 0.5, "Baseline Image", fontsize=14)
    ax_text.axis('off')
    
    # Rows 1..N: Attacks
    for i, (name, res) in enumerate(results.items()):
        row = i + 1
        
        adv_tensor = res["image"]
        pred_class = res["pred"]
        
        # Calculate Noise
        noise = (adv_tensor - img).abs()
        # Amplify noise for visibility (x10)
        viz_noise = noise * 10
        
        L2 = (adv_tensor - img).norm().item()
        
        # Plot Adversarial Image
        ax_img = axes[row, 0]
        ax_img.imshow(denorm(adv_tensor).cpu().squeeze().permute(1, 2, 0))
        color = "red" if res["success"] else "green"
        ax_img.set_title(f"{name}\nPred: {pred_class}", color=color, fontweight='bold')
        ax_img.axis('off')
        
        # Plot Noise
        ax_noise = axes[row, 1]
        ax_noise.imshow(denorm(viz_noise).cpu().squeeze().permute(1, 2, 0), cmap='hot')
        ax_noise.set_title("Noise (x10 amplified)")
        ax_noise.axis('off')
        
        # details
        ax_text = axes[row, 2]
        info = f"Success: {res['success']}\nL2 Distortion: {L2:.4f}\n"
        if name == "AutoAttack":
            info += "Type: White-Box (PGD+APGD)"
        elif name == "DeepFool":
            info += "Type: White-Box (Min-Norm)"
        elif name == "Boundary":
            info += "Type: Black-Box (Decision-based)"
            
        ax_text.text(0.1, 0.4, info, fontsize=12)
        ax_text.axis('off')

    plt.tight_layout()
    save_path = os.path.join(config.DATA_DIR, "viz_results", "attack_gallery.png")
    plt.savefig(save_path)
    print(f"Gallery saved to {save_path}")

if __name__ == "__main__":
    visualize_all()
