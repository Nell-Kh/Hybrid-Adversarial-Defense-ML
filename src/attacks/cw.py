
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Adjust path to enable absolute imports if running as script
if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from src.attacks.base import Attacker

class CWAttacker(Attacker):
    """
    Carlini-Wagner (C&W) L2 Attack (The Sniper).
    
    Paper: "Towards Evaluating the Robustness of Neural Networks" (2017)
    
    This attack optimizes:
        Minimize ||delta||_2^2 + c * f(x + delta)
        
    Where f() is a function that is <= 0 if the image is misclassified, 
    and > 0 if it is correctly classified.
    
    It uses a change-of-variable (tanh) to ensure pixels stay in [-1, 1].
    """
    def __init__(self, model, device, c=1.0, kappa=0, steps=100, lr=0.01):
        """
        Args:
            c: Trade-off constant (Higher = stronger attack, more noise).
            kappa: Confidence margin (0 = just flip label, >0 = flip with high confidence).
            steps: Optimization steps (100 is a "Lite" version, paper uses 10,000).
            lr: Learning Rate for Adam optimizer.
        """
        super().__init__(model, device)
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr

    def attack(self, images, labels, target_labels=None):
        images = images.clone().detach().to(self.device)
        labels = labels.to(self.device)
        if target_labels is not None:
            target_labels = target_labels.to(self.device)
        
        # C&W uses a "Change of Variable" to handle the box constraints [-1, 1]
        # Instead of optimizing 'delta', we optimize 'w'.
        # adv_image = tanh(w)
        # To start with the original image: w = arctanh(image)
        # Note: images must be strictly in (-1, 1) for arctanh, so we clip slightly
        w = torch.atanh(images * 0.9999).detach()
        w.requires_grad = True
        
        optimizer = optim.Adam([w], lr=self.lr)
        
        best_adv_images = images.clone().detach()
        best_l2 = float('inf') * torch.ones(images.size(0)).to(self.device)
        
        print(f"Running CW Attack (Steps={self.steps}, c={self.c})...")
        
        for step in range(self.steps):
            # 1. Forward Pass
            # Convert w back to image space: tanh(w) -> [-1, 1]
            adv_images = torch.tanh(w)
            
            # 2. Calculate Loss
            # Ref: https://arxiv.org/pdf/1608.04644.pdf (Eq 6)
            
            # A) L2 Distance Loss (Minimize noise)
            l2_dist = torch.sum((adv_images - images)**2, dim=[1, 2, 3])
            
            # B) Classification Loss (Force wrong label)
            outputs = self.model(adv_images)
            
            if target_labels is None:
                # UNTARGETED: We want real class to drop below the next highest class
                real_score = torch.gather(outputs, 1, labels.unsqueeze(1)).squeeze(1)
                tmp_outputs = outputs.clone()
                one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=outputs.shape[1])
                tmp_outputs = tmp_outputs - one_hot_labels * 10000.0
                other_score, _ = torch.max(tmp_outputs, dim=1)
                # Maximize other_score, minimize real_score
                f_loss = torch.clamp(real_score - other_score + self.kappa, min=0)
            else:
                # TARGETED: We want the target class to be higher than ANY other class
                target_score = torch.gather(outputs, 1, target_labels.unsqueeze(1)).squeeze(1)
                tmp_outputs = outputs.clone()
                one_hot_targets = torch.nn.functional.one_hot(target_labels, num_classes=outputs.shape[1])
                tmp_outputs = tmp_outputs - one_hot_targets * 10000.0
                other_score, _ = torch.max(tmp_outputs, dim=1)
                # Maximize target_score, minimize other_score
                f_loss = torch.clamp(other_score - target_score + self.kappa, min=0)
            
            # Total Loss
            cost = l2_dist + self.c * f_loss
            
            optimizer.zero_grad()
            cost.sum().backward()
            optimizer.step()
            
            # 3. Save Best Result
            with torch.no_grad():
                pred = outputs.argmax(1)
                
                if target_labels is None:
                    mask_success = (pred != labels) # True if untargeted attack succeeded
                else:
                    mask_success = (pred == target_labels) # True if targeted attack succeeded
                
                # Update if successful AND lower L2 distance than before
                update_idx = mask_success & (l2_dist < best_l2)
                
                best_adv_images[update_idx] = adv_images[update_idx]
                best_l2[update_idx] = l2_dist[update_idx]
        
        return best_adv_images

if __name__ == "__main__":
    # Test Run
    from src.model import get_model
    from src.dataset import get_dataloaders
    import src.config as config
    
    device = config.DEVICE
    model = get_model(device)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    
    # Load Data
    train_loader, _ = get_dataloaders(batch_size=4)
    images, labels = next(iter(train_loader))
    
    # Attack
    print("\n--- Testing Carlini-Wagner (The Sniper) ---")
    attacker = CWAttacker(model, device, steps=100, c=1.0)
    adv_imgs = attacker.attack(images, labels)
    
    # Check Logic
    clean_preds = model(images.to(device)).argmax(1)
    adv_preds = model(adv_imgs).argmax(1)
    
    print("\n=== RESULTS ===")
    print(f"True Labels : {labels.tolist()}")
    print(f"Clean Preds : {clean_preds.tolist()}")
    print(f"Adv Preds   : {adv_preds.tolist()}")
    
    success = (adv_preds != labels.to(device)).float().mean()
    print(f"Success Rate: {success.item()*100:.1f}%")
