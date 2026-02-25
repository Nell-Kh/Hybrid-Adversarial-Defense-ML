"""
[FILE PURPOSE]
This file implements "AutoAttack Lite" (The Tank).
It is a WHITE-BOX attack (needs gradients).

[THE GOAL]
This is the standard "Stress Test" for any defense.
If you claim your model is safe, you must pass AutoAttack.
It is designed to be reliable—it tries multiple strategies to make sure it finds a weakness if one exists.

[HOW IT WORKS]
It runs an Ensemble (a team) of attacks:
1. PGD with Cross-Entropy Loss: The standard approach. Tries to maximize the error.
2. PGD with DLR Loss: A specialized mathematical loss that ignores the total score and focuses only on the gap between "Top 1" and "Top 2" classes.
3. It takes the WORST result (if either one succeeds, the image is considered broken).

[WHAT IT NEEDS]
- High computational power (it runs the model many times).
"""

from .base import Attacker
import torch
import torch.nn as nn
import src.config as config

class AutoAttackLite(Attacker):
    def __init__(self, model, device, eps=config.ATTACK_EPSILON, alpha=config.ATTACK_ALPHA, steps=config.ATTACK_STEPS, restarts=3):
        super().__init__(model, device)
        # Epsilon: The maximum amount of noise allowed (The "Power" of the attack)
        self.eps = eps
        # Alpha: The step size for each iteration
        self.alpha = alpha
        # Steps: How many times we try to improve the attack
        self.steps = steps
        # Restarts: How many random starting points we try to avoid local minima
        self.restarts = restarts

    def dlr_loss_batched(self, outputs, labels):
        """
        Batched Difference of Logits Ratio (DLR) Loss.
        Returns loss per image.
        """
        logits_sorted, logits_idx = outputs.sort(dim=1, descending=True)
        top1_logits = logits_sorted[:, 0]
        top2_logits = logits_sorted[:, 1]
        top1_idx = logits_idx[:, 0]
        
        true_logits = outputs.gather(1, labels.unsqueeze(1)).squeeze(1)
        other_logits = torch.where(labels == top1_idx, top2_logits, top1_logits)
        
        return -(true_logits - other_logits)

    def dlr_loss(self, outputs, labels):
        return self.dlr_loss_batched(outputs, labels).sum()

    def run_pgd(self, images: torch.Tensor, labels: torch.Tensor, loss_type: str="ce") -> torch.Tensor:
        """
        Projected Gradient Descent (PGD) with multiple restarts.
        """
        best_adv = images.clone().detach()
        # Worst loss -> we want to find the perturbation that MAXIMIZES loss
        max_loss = -float('inf') * torch.ones(images.size(0)).to(self.device)
        
        loss_fn = nn.CrossEntropyLoss(reduction='none')

        for restart in range(self.restarts):
            adv_images = images.clone().detach()
            # Random uniform start in [-eps, eps]
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, -1, 1).to(self.device)
            
            for _ in range(self.steps):
                adv_images.requires_grad = True
                outputs = self.model(adv_images)
                
                if loss_type == "ce":
                    loss_batch = loss_fn(outputs, labels)
                    loss = loss_batch.sum()
                elif loss_type == "dlr":
                    loss_batch = self.dlr_loss_batched(outputs, labels)
                    loss = loss_batch.sum()
                    
                grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
                
                adv_images = adv_images.detach() + self.alpha * grad.sign()
                delta = torch.clamp(adv_images - images.to(self.device), min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images.to(self.device) + delta, min=-1, max=1).detach()
                
            # Final evaluation for this restart
            with torch.no_grad():
                final_outputs = self.model(adv_images)
                if loss_type == "ce":
                    final_loss = loss_fn(final_outputs, labels)
                else:
                    final_loss = self.dlr_loss_batched(final_outputs, labels)
                
                update_mask = final_loss > max_loss
                max_loss[update_mask] = final_loss[update_mask]
                best_adv[update_mask] = adv_images[update_mask]
                
        return best_adv

    def attack(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # 1. Run Strategy A (Cross-Entropy)
        adv_ce = self.run_pgd(images, labels, loss_type="ce")
        
        # 2. Run Strategy B (DLR)
        adv_dlr = self.run_pgd(images, labels, loss_type="dlr")
        
        # 3. Combine Results
        # We want to be a Perfectionist Villain.
        # If Strategy A worked, great. If not, check Strategy B.
        
        with torch.no_grad():
             pred_ce = self.model(adv_ce).argmax(1)
             pred_dlr = self.model(adv_dlr).argmax(1)
        
        final_adv = adv_ce.clone()
        
        # Logic: If CE failed (Model was right) BUT DLR succeeded (Model was wrong), switch to DLR.
        ce_failed = (pred_ce == labels)
        dlr_success = (pred_dlr != labels)
        swap_mask = ce_failed & dlr_success
        
        final_adv[swap_mask] = adv_dlr[swap_mask]
        return final_adv
