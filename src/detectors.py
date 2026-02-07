import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from sklearn.metrics import roc_auc_score
import config
from model import get_model
from dataset import get_dataloaders
from auto_attack import AutoAttackLite
from tqdm import tqdm
import os

# --- PREDICTION STABILITY DETECTOR ---
# Concept: "Fragile" Adversarial Examples vs "Robust" Real Images
# Real images retain their identity under aggressive transformations (Resize, JPEG).
# Adversarial noise is high-frequency and specific; it breaks under these transforms.

IMG_SIZE = 64
RESIZE_TARGET = 28  # Crush to 28x28 (MNIST size)
JPEG_QUALITY = 15   # Heavy compression

def get_kl_divergence(logits1, logits2):
    # Measures how different two probability distributions are
    p = F.log_softmax(logits1, dim=1)
    q = F.softmax(logits2, dim=1)
    return F.kl_div(p, q, reduction='batchmean').item()

# --- TRANSFORM 1: AGGRESSIVE RESIZING ---
def transform_aggressive_resize(image):
    # Differentiable approximation of resizing
    # Input assumed to be normalized (-1, 1) approx
    
    # 1. Denormalize to (0, 1) for interpolation math
    img = image * 0.5 + 0.5
    
    # 2. Resize Down
    down = F.interpolate(img, size=RESIZE_TARGET, mode='bilinear', align_corners=False)
    
    # 3. Resize Up
    up = F.interpolate(down, size=IMG_SIZE, mode='bilinear', align_corners=False)
    
    # 4. Renormalize to (-1, 1)
    return (up - 0.5) / 0.5

# --- TRANSFORM 2: JPEG COMPRESSION (Simulation) ---
# Differentiable JPEG is hard, so we use the non-differentiable PIL version for inference/eval
# or a simple quantization approximation.
# For this detector, we don't need gradients on the defense, so PIL is fine.
def transform_aggressive_jpeg(image):
    # Image is (B, 3, 64, 64) standard tensor
    device = image.device
    
    # Move to CPU for PIL
    # We process item by item or batch if we had a batch-jpeg-lib (rare)
    # Item by item for safety
    out_batch = []
    
    for i in range(image.size(0)):
        img_t = image[i].cpu()
        
        # Denorm
        img_t = img_t * 0.5 + 0.5
        img_t = torch.clamp(img_t, 0, 1)
        
        to_pil = transforms.ToPILImage()
        pil_img = to_pil(img_t)
        
        # Save to buffer
        import io
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=JPEG_QUALITY)
        buffer.seek(0)
        jpeg_img = transforms.ToTensor()(torch.load(buffer) if False else  from_pil(buffer)) # psuedocode fix below
        
        # Actual PIL Load
        from PIL import Image
        jpeg_img = Image.open(buffer)
        t_img = transforms.ToTensor()(jpeg_img)
        
        # Norm
        t_img = (t_img - 0.5) / 0.5
        out_batch.append(t_img)
        
    return torch.stack(out_batch).to(device)

def from_pil(buffer):
    from PIL import Image
    return Image.open(buffer)

class StabilityDetector:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def score(self, images):
        self.model.eval()
        
        # 1. Original Prediction
        with torch.no_grad():
            orig_logits = self.model(images)
            
        # 2. Resized Prediction
        img_resized = transform_aggressive_resize(images)
        with torch.no_grad():
            resize_logits = self.model(img_resized)
            
        # 3. JPEG Prediction
        img_jpeg = transform_aggressive_jpeg(images)
        with torch.no_grad():
            jpeg_logits = self.model(img_jpeg)
            
        # 4. Calculate Scores (batch-wise)
        scores = []
        for i in range(len(images)):
            # KL Div expects batch, so unsqueeze
            d1 = get_kl_divergence(resize_logits[i].unsqueeze(0), orig_logits[i].unsqueeze(0))
            d2 = get_kl_divergence(jpeg_logits[i].unsqueeze(0), orig_logits[i].unsqueeze(0))
            
            # Anomaly Score = Max divergence
            scores.append(max(d1, d2))
            
        return np.array(scores)

def evaluate_stability():
    device = config.DEVICE
    print(f"Running Stability Detector (Physics Mode) on {device}...")
    
    model = get_model(device)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    model.eval()
    
    detector = StabilityDetector(model, device)
    attacker = AutoAttackLite(model, device)
    
    # Load Data
    _, val_loader = get_dataloaders()
    
    y_true = []
    y_scores = []
    
    count = 0
    MAX_TEST = 50
    
    for images, labels in tqdm(val_loader, desc="Testing"):
        if count >= MAX_TEST: break
        
        images, labels = images.to(device), labels.to(device)
        
        # A. Clean Images
        scores_clean = detector.score(images)
        y_true.extend([0] * len(images))
        y_scores.extend(scores_clean)
        
        # B. Attack Images
        # Filter correct only
        with torch.no_grad(): preds = model(images).argmax(1)
        mask = preds == labels
        if not mask.any(): continue
        
        img_atk = images[mask]
        lbl_atk = labels[mask]
        
        adv_images = attacker.attack(img_atk, lbl_atk)
        
        scores_adv = detector.score(adv_images)
        y_true.extend([1] * len(adv_images))
        y_scores.extend(scores_adv)
        
        count += len(images)
        
    auc = roc_auc_score(y_true, y_scores)
    
    # Calculate Means
    real_scores = [s for i,s in zip(y_true, y_scores) if i==0]
    fake_scores = [s for i,s in zip(y_true, y_scores) if i==1]
    
    print(f"\n\n--- RESULTS (Stability Detector) ---")
    print(f"ROC-AUC Score: {auc:.4f}")
    print(f"Avg Instability (Real): {np.mean(real_scores):.4f}")
    print(f"Avg Instability (Fake): {np.mean(fake_scores):.4f}")
    
    if auc > 0.70:
        print("Verdict: SUCCESS. Physics wins.")
    else:
        print("Verdict: The model is too unstable even on clean images.")

if __name__ == "__main__":
    evaluate_stability()
