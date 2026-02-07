import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import argparse
from model import get_model
from dataset import get_dataloaders
from patch_attack import PatchApplier, AdversarialPatch # Re-using classes
import config

def get_saliency_map(model, image, label):
    """
    Computes the gradient of the score of class 'label' w.r.t the input image.
    """
    image.requires_grad = True
    outputs = model(image)
    
    score = outputs[0][label]
    score.backward()
    
    # Saliency is the max magnitude across channels
    saliency, _ = torch.max(image.grad.data.abs(), dim=1)
    return saliency

def visualize_impact(patch_path=os.path.join(config.DATA_DIR, "patch_attack", "universal_patch.pt")):
    device = config.DEVICE
    print(f"Visualizing Patch Impact on {device}...")
    
    # Load Model
    model = get_model(device)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    model.eval()
    
    # Load Patch
    if not os.path.exists(patch_path):
        print(f"Error: Patch file not found at {patch_path}")
        return

    patch_tensor = torch.load(patch_path, map_location=device)
    
    # Mock Patch Object for Applier
    # We create a dummy module to hold the patch tensor
    class FixedPatch(torch.nn.Module):
        def __init__(self, tensor):
            super().__init__()
            self.patch = tensor
        def forward(self):
            return self.patch
            
    patch_module = FixedPatch(patch_tensor)
    applier = PatchApplier(device, img_size=64, min_scale=0.3, max_scale=0.4) # Fixed scale for nice viz
    
    # Load Data
    _, val_loader = get_dataloaders()
    images, labels = next(iter(val_loader))
    images, labels = images.to(device), labels.to(device)
    
    # Pick a few samples
    num_samples = 3
    # Find images that are NOT the target (assuming target is 0)
    # We want to show how we FLIP the prediction
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(12, 3*num_samples))
    # Columns: Original, Original Saliency, Patched, Patched Saliency
    
    count = 0
    for i in range(len(images)):
        if count >= num_samples: break
        
        img = images[i:i+1] # Keep batch dim
        label = labels[i]
        
        # 1. Original Prediction & Saliency
        orig_pred = model(img).argmax(1).item()
        orig_saliency = get_saliency_map(model, img, orig_pred)
        
        # 2. Apply Patch
        # We need to zero grad because get_saliency_map messes with it
        model.zero_grad()
        img.grad = None
        img.requires_grad = False
        
        patched_img = applier(img, patch_module.patch)
        patched_pred = model(patched_img).argmax(1).item()
        
        # 3. Patched Saliency
        # Saliency w.r.t to the NEW prediction (to see where it's looking)
        patched_saliency = get_saliency_map(model, patched_img.detach(), patched_pred)
        
        # Plotting
        ax = axes[count]
        
        # Helper to plot tensor
        def show(ax, t, title):
            t = t.cpu().squeeze().detach()
            if t.dim() == 3: # Image
                t = t * 0.5 + 0.5 # Denorm
                t = t.clamp(0, 1)
                ax.imshow(t.permute(1, 2, 0))
            else: # Saliency
                ax.imshow(t, cmap='hot')
            ax.set_title(title, fontsize=10)
            ax.axis('off')
            
        show(ax[0], img, f"Original\nPred: {orig_pred}")
        show(ax[1], orig_saliency, "Original Attention")
        show(ax[2], patched_img, f"Patched\nPred: {patched_pred}")
        show(ax[3], patched_saliency, "Patched Attention")
        
        count += 1
        
    plt.tight_layout()
    output_file = os.path.join(config.DATA_DIR, "patch_attack", "saliency_viz.png")
    plt.savefig(output_file)
    print(f"Visualization saved to {output_file}")

if __name__ == "__main__":
    visualize_impact()
