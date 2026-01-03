import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from model import get_model
from dataset import get_dataloaders
from tqdm import tqdm
import os
from torchvision.utils import save_image

# --- AUTO ATTACK IMPLEMENTATION (Lite Version) ---
# Research Paper: "Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks" (ICML 2020)
# We implement an ensemble of:
# 1. PGD with CrossEntropy Loss (Standard)
# 2. PGD with DLR Loss (Difference of Logits Ratio) - Targets the top-2 classes separation
# 3. FGSM (Fast Gradient Sign Method) - Quick and dirty

class AutoAttackLite:
    def __init__(self, model, device, eps=config.ATTACK_EPSILON, alpha=config.ATTACK_ALPHA, steps=config.ATTACK_STEPS):
        self.model = model
        self.device = device
        self.eps = eps
        self.alpha = alpha
        self.steps = steps

    def dlr_loss(self, outputs, labels):
        # Difference of Logits Ratio Loss (simplified)
        # Goal: Minimize the difference between the true class logit and the next highest logit
        
        # Sort logits to find top 2
        logits_sorted, logits_idx = outputs.sort(dim=1, descending=True)
        
        # If true class is top 1, take difference with top 2. 
        # If true class is NOT top 1, take difference with top 1.
        
        # Get true class logits
        # gather: selection using index
        true_logits = outputs.gather(1, labels.unsqueeze(1)).squeeze()
        
        # Find the max other logit
        # We can just mask the true class and take max
        # A simpler stable way for implementation:
        # Take max(top1, top2). If label==top1, use top2, else use top1
        
        top1_logits = logits_sorted[:, 0]
        top2_logits = logits_sorted[:, 1]
        
        # Check if label is top1
        # indices of top1
        top1_idx = logits_idx[:, 0]
        
        # If label is top1, we want top2. Else top1.
        other_logits = torch.where(labels == top1_idx, top2_logits, top1_logits)
        
        # DLR = -(True - Other) / (Top - Bottom) ... ignoring denominator for simple APGD
        # We want to minimize (True - Other). 
        # maximizing difference -> making clean prediction confident
        # minimizing difference -> causing misclassification
        
        return -(true_logits - other_logits).sum()

    def run_pgd(self, images, labels, loss_type="ce"):
        adv_images = images.clone().detach()
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
        adv_images = torch.clamp(adv_images, 0, 1).to(self.device)
        
        loss_fn = nn.CrossEntropyLoss()
        
        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)
            
            if loss_type == "ce":
                loss = loss_fn(outputs, labels)
            elif loss_type == "dlr":
                loss = self.dlr_loss(outputs, labels) # This minimizes the margin
                
            grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
            
            # Maximize Logit Difference (DLR) or CE Loss
            # CE Loss: Maximize loss to find adv
            # DLR Loss above: returns -(True-Other). Minimizing this = maximizing (True-Other).
            # Wait, for attack we want to MINIMIZE (True - Other).
            # So if DLR returns -(True - Other), we want to MAXIMIZE output.
            
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            
            delta = torch.clamp(adv_images - images.to(self.device), min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images.to(self.device) + delta, min=0, max=1).detach()
            
        return adv_images

    def attack(self, images, labels):
        # 1. Run CE-PGD
        adv_ce = self.run_pgd(images, labels, loss_type="ce")
        
        # 2. Run DLR-PGD
        adv_dlr = self.run_pgd(images, labels, loss_type="dlr")
        
        # 3. Check which one worked better (caused error)
        # We select the one that misclassifies. If both misclassify, pick the one with higher loss?
        # A simple AutoAttack strategy:
        # Check if CE fooled it. If yes, keep it.
        # If not, check if DLR fooled it. If yes, take DLR.
        # If neither, take CE (best effort).
        
        with torch.no_grad():
             pred_ce = self.model(adv_ce).argmax(1)
             pred_dlr = self.model(adv_dlr).argmax(1)
        
        final_adv = adv_ce.clone()
        
        # Where CE failed (pred_ce == labels) AND DLR succeeded (pred_dlr != labels), swap to DLR
        ce_failed = (pred_ce == labels)
        dlr_success = (pred_dlr != labels)
        swap_mask = ce_failed & dlr_success
        
        final_adv[swap_mask] = adv_dlr[swap_mask]
        
        return final_adv

def generate_strong_attacks():
    device = config.DEVICE
    print(f"Initializing AutoAttack on {device}...")
    
    model = get_model(device)
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print("Error: Train model first.")
        return
        
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    model.eval()
    
    attacker = AutoAttackLite(model, device)
    
    # Load Data
    _, val_loader = get_dataloaders()
    
    # Storage
    saved_count = 0
    target_count = 50
    output_dir = os.path.join(config.DATA_DIR, "auto_attack_samples")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating AutoAttack samples...")
    
    for images, labels in val_loader:
        if saved_count >= target_count: break
        
        images, labels = images.to(device), labels.to(device)
        
        # Only attack correctly classified images
        with torch.no_grad():
            preds = model(images).argmax(1)
        mask = preds == labels
        if not mask.any(): continue
        
        images = images[mask]
        labels = labels[mask]
        
        # Run Attack
        adv_images = attacker.attack(images, labels)
        
        # Verify Success
        with torch.no_grad():
            adv_preds = model(adv_images).argmax(1)
            
        success_mask = adv_preds != labels
        
        # Save successful attacks
        for i in range(len(images)):
            if saved_count >= target_count: break
            if success_mask[i]:
                save_image(adv_images[i], os.path.join(output_dir, f"adv_{saved_count}.png"))
                saved_count += 1
                if saved_count % 10 == 0:
                     print(f"Generated {saved_count}/{target_count} strong attacks.")

if __name__ == "__main__":
    generate_strong_attacks()
