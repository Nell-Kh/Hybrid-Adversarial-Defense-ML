import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from model import get_model
from dataset import get_dataloaders
from attack import pgd_attack
import cv2

def get_gradcam(model, image, label, device):
    """
    Computes the Grad-CAM heatmap for a specific class.
    """
    model.eval()
    
    # Hook variables
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
        
    def forward_hook(module, input, output):
        activations.append(output)
        
    # Hook into the last convolutional layer (layer4)
    target_layer = model.layer4[-1]
    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)
    
    # Forward Pass
    output = model(image)
    
    # Backward Pass (Targeting the specific class)
    model.zero_grad()
    score = output[0, label]
    score.backward()
    
    # Generate Heatmap
    grads = gradients[0].cpu().data.numpy()[0] # (512, 2, 2)
    fmap = activations[0].cpu().data.numpy()[0] # (512, 2, 2)
    
    # Global Average Pooling of gradients
    weights = np.mean(grads, axis=(1, 2))
    
    # Weighted combination of feature maps
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * fmap[i]
        
    # ReLU
    cam = np.maximum(cam, 0)
    
    # Resize to image size (64x64)
    cam = cv2.resize(cam, (64, 64))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam) # Normalize 0-1
    
    handle_f.remove()
    handle_b.remove()
    
    return cam

def show_visuals():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Generating Grad-CAM on {device}...")
    
    model = get_model(device)
    try:
        model.load_state_dict(torch.load("../models/resnet_tinyimagenet.pth", map_location=device))
    except:
        print("Model not found! Run train.py first.")
        return

    loader = get_dataloaders()
    images, labels = next(iter(loader))
    
    # Use first image
    img = images[0].unsqueeze(0).to(device)
    label = labels[0]
    
    # 1. Clean Grad-CAM
    heatmap_clean = get_gradcam(model, img, label, device)
    
    # 2. Generate Attack
    adv_img = pgd_attack(model, img, label.unsqueeze(0), device)
    heatmap_adv = get_gradcam(model, adv_img, label, device)
    
    # Prepare plotting
    img_cpu = (img.squeeze().permute(1,2,0).cpu().numpy() * 0.5 + 0.5)
    adv_cpu = (adv_img.detach().squeeze().permute(1,2,0).cpu().numpy() * 0.5 + 0.5)
    
    plt.figure(figsize=(10, 5))
    
    # Plot Clean
    plt.subplot(1, 2, 1)
    plt.imshow(img_cpu)
    plt.imshow(heatmap_clean, cmap='jet', alpha=0.5)
    plt.title("Clean Image Attention")
    plt.axis('off')
    
    # Plot Adversarial
    plt.subplot(1, 2, 2)
    plt.imshow(adv_cpu)
    plt.imshow(heatmap_adv, cmap='jet', alpha=0.5)
    plt.title("Adversarial Image Attention")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_visuals()