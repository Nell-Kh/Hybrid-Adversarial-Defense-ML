import torch
import torch.nn.functional as F
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from sklearn.metrics import roc_auc_score
from model import get_model

# Configuration
IMG_SIZE = 64
NOISE_STD = 0.02
BLUR_KERNEL = 3
ROT_DEGREES = 5

def load_image(path, device):
    t = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    try:
        img = Image.open(path).convert('RGB')
        return t(img).unsqueeze(0).to(device)
    except:
        return None

def get_kl_divergence(logits1, logits2):
    """
    Calculates the Kullback-Leibler Divergence between two probability distributions.
    This is a professional metric for measuring prediction instability.
    """
    p = F.log_softmax(logits1, dim=1)
    q = F.softmax(logits2, dim=1)
    return F.kl_div(p, q, reduction='batchmean').item()

def get_stability_score(model, image):
    """
    Detector 1: Prediction Stability Analysis.
    Measures how much the model's confidence changes under stress.
    """
    model.eval()
    
    # 1. Baseline Prediction
    with torch.no_grad():
        orig_logits = model(image)

    # 2. Stress Test A: Gaussian Noise
    noise = torch.randn_like(image) * NOISE_STD
    with torch.no_grad():
        noise_logits = model(image + noise)
        
    # 3. Stress Test B: Gaussian Blur
    img_blur = TF.gaussian_blur(image, BLUR_KERNEL)
    with torch.no_grad():
        blur_logits = model(img_blur)
        
    # 4. Stress Test C: Geometric Rotation
    img_rot = TF.rotate(image, ROT_DEGREES)
    with torch.no_grad():
        rot_logits = model(img_rot)

    # Calculate Instability (Distance from original)
    d1 = get_kl_divergence(noise_logits, orig_logits)
    d2 = get_kl_divergence(blur_logits, orig_logits)
    d3 = get_kl_divergence(rot_logits, orig_logits)
    
    # The Anomaly Score is the maximum instability found
    return max(d1, d2, d3)

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running Stability Detector (KL-Divergence) on {device}...")

    model = get_model(device)
    model.load_state_dict(torch.load("../models/resnet_tinyimagenet.pth", map_location=device))
    model.eval()

    clean_dir = "../data/clean_samples"
    adv_dir = "../data/adversarial_samples"
    
    y_true = []
    y_scores = []
    
    count = 0
    print("Processing images...", end="", flush=True)
    
    while True:
        c_path = os.path.join(clean_dir, f"img_{count}.png")
        a_path = os.path.join(adv_dir, f"img_{count}.png")
        
        if not os.path.exists(c_path):
            break
            
        if count % 20 == 0:
            print(".", end="", flush=True)
        
        # Test Clean
        s_c = get_stability_score(model, load_image(c_path, device))
        y_true.append(0) # 0 = Real
        y_scores.append(s_c)
        
        # Test Fake
        s_a = get_stability_score(model, load_image(a_path, device))
        y_true.append(1) # 1 = Fake
        y_scores.append(s_a)
        
        count += 1

    # Evaluation
    auc = roc_auc_score(y_true, y_scores)
    
    print(f"\n\n--- Results (N={count*2}) ---")
    print(f"ROC-AUC Score: {auc:.4f}")
    
    real_mean = np.mean([s for i,s in zip(y_true, y_scores) if i==0])
    fake_mean = np.mean([s for i,s in zip(y_true, y_scores) if i==1])
    
    print(f"Mean Instability (Real): {real_mean:.4f}")
    print(f"Mean Instability (Fake): {fake_mean:.4f}")

if __name__ == "__main__":
    main()