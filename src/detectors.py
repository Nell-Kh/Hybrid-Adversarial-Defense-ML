import torch
import torch.nn.functional as F
import os
import io
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from model import get_model

# --- AGGRESSIVE CONFIGURATION ---
IMG_SIZE = 64
# We shrink the image to tiny 28x28 (MNIST size)
# This forces the attack pixels to merge and disappear
RESIZE_TARGET = 28  
# We use very low quality JPEG (15) to crush high-freq noise
JPEG_QUALITY = 15   

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
    p = F.log_softmax(logits1, dim=1)
    q = F.softmax(logits2, dim=1)
    return F.kl_div(p, q, reduction='batchmean').item()

# --- TRANSFORM 1: AGGRESSIVE RESIZING ---
def transform_aggressive_resize(image, device):
    img = image * 0.5 + 0.5
    # Shrink to tiny size
    down = F.interpolate(img, size=RESIZE_TARGET, mode='bilinear', align_corners=False)
    # Stretch back
    up = F.interpolate(down, size=IMG_SIZE, mode='bilinear', align_corners=False)
    return (up - 0.5) / 0.5

# --- TRANSFORM 2: DEEP JPEG COMPRESSION ---
def transform_aggressive_jpeg(image, device):
    img = image.clone() * 0.5 + 0.5
    img = torch.clamp(img, 0, 1)
    to_pil = transforms.ToPILImage()
    pil_img = to_pil(img.squeeze(0).cpu())
    
    buffer = io.BytesIO()
    # Quality 15 is very blocky
    pil_img.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    buffer.seek(0)
    jpeg_img = Image.open(buffer)
    
    to_tensor = transforms.ToTensor()
    t_img = to_tensor(jpeg_img).unsqueeze(0).to(device)
    return (t_img - 0.5) / 0.5

def get_anomaly_score(model, image, device):
    model.eval()
    
    # 1. Original Prediction
    with torch.no_grad():
        orig_logits = model(image)

    # 2. Run Aggressive Transforms
    img_resized = transform_aggressive_resize(image, device)
    img_jpeg = transform_aggressive_jpeg(image, device)
    
    with torch.no_grad():
        resize_logits = model(img_resized)
        jpeg_logits = model(img_jpeg)

    # 3. Measure Instability
    # If the image is Real, it should still look like a "Blurry Bird" (Low distance)
    # If the image is Fake, the "Spider" pattern should be gone (High distance)
    d1 = get_kl_divergence(resize_logits, orig_logits)
    d2 = get_kl_divergence(jpeg_logits, orig_logits)
    
    return max(d1, d2)

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running Aggressive Detector (Resize={RESIZE_TARGET}, JPEG={JPEG_QUALITY}) on {device}...")

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
        
        if not os.path.exists(c_path): break
        if count % 10 == 0: print(".", end="", flush=True)

        s_c = get_anomaly_score(model, load_image(c_path, device), device)
        y_true.append(0)
        y_scores.append(s_c)
        
        s_a = get_anomaly_score(model, load_image(a_path, device), device)
        y_true.append(1)
        y_scores.append(s_a)
        
        count += 1

    auc = roc_auc_score(y_true, y_scores)
    
    print(f"\n\nResults (N={count*2})")
    print(f"ROC-AUC Score: {auc:.4f}")
    
    real_mean = np.mean([s for i,s in zip(y_true, y_scores) if i==0])
    fake_mean = np.mean([s for i,s in zip(y_true, y_scores) if i==1])
    
    print(f"Avg Instability (Real): {real_mean:.4f}")
    print(f"Avg Instability (Fake): {fake_mean:.4f}")

if __name__ == "__main__":
    main()