"""
[FILE PURPOSE]
This file implements the "Boundary Attack" (The Hacker).
It is a BLACK-BOX attack. This means it breaks the model WITHOUT knowing the math inside it (no gradients).
It simulates a rigorous "Hacker" scenario where you only get to ask the API "What is this?"

[HOW IT WORKS]
1. Start with an image that is ALREADY successful (e.g., a picture of a truck).
2. The Model says "Truck" (Success). But we want a "Dog" to look like a "Truck".
3. We take a picture of a Dog (Original).
4. We slowly move the "Truck" image towards the "Dog" image, step by step.
5. At every step, we check: "Do you still think this is a Truck?"
6. If yes, we keep moving closer to the Dog.
7. Eventually, we get an image that looks almost exactly like a Dog, but the model still swears it's a Truck.

[WHAT IT NEEDS]
- Ideally, a better initialization strategy (currently it just picks a random image).
- But for this project, it is fully functional.
"""

from .base import Attacker
import torch
import numpy as np

class BoundaryAttack(Attacker):
    def __init__(self, model, device, steps=5000, spherical_step=0.01, source_step=0.01):
        super().__init__(model, device)
        self.steps = steps
        
        # How much we wiggle sideways (exploring the edge)
        self.spherical_step = spherical_step 
        
        # How much we push towards the original image (becoming invisible)
        self.source_step = source_step       
        
    def _is_adversarial(self, perturbed, original_labels):
        """
        Asks the model: "Did I fool you?"
        Returns True if the prediction is WRONG (Success).
        """
        with torch.no_grad():
            preds = self.model(perturbed).argmax(1)
        return preds != original_labels

    def attack(self, images, labels):
        batch_size = images.size(0)
        
        # --- PHASE 1: INITIALIZATION ---
        # We need to start with ANY image that the model gets wrong.
        # In a real attack, you'd pick a specific target.
        # Here, we just pick random noise or other images until one works.
        
        adv_images = images.clone()
        
        # Find a starting point for each image in the batch
        for i in range(batch_size):
            while True:
                # Try a random image (random noise for simplicity)
                init_adv = torch.rand_like(images[i]).unsqueeze(0).to(self.device)
                
                # If this random noise is technically "Not a Dog", then it's a valid start!
                if self._is_adversarial(init_adv, labels[i:i+1]):
                    adv_images[i] = init_adv.squeeze(0)
                    break
        
        # --- PHASE 2: THE RANDOM Walk ---
        curr_adv = adv_images.clone()
        
        for step in range(self.steps):
            # Calculate the path from where we are (Bad) to where we want to be (The Original Image)
            diff = images - curr_adv 
            
            # 1. Take a step towards the Original Image
            # This makes the attack look less noisy and more real.
            candidate = curr_adv + self.source_step * diff
            
            # 2. Add some random noise sideways
            # This helps us "explore" the decision boundary curve.
            noise = torch.randn_like(candidate).to(self.device)
            
            # (Math Magic): Make the noise strictly orthogonal (perpendicular) to the direction we are moving.
            # This ensures we don't accidentally undo our progress towards the original.
            diff_flat = diff.view(batch_size, -1)
            noise_flat = noise.view(batch_size, -1)
            
            # Project the noise
            proj_factor = (noise_flat * diff_flat).sum(1, keepdim=True) / (diff_flat * diff_flat).sum(1, keepdim=True) + 1e-12
            proj_noise = noise_flat - proj_factor * diff_flat
            proj_noise = proj_noise.view_as(curr_adv)
            
            # Normalize the noise size
            proj_norm = proj_noise.view(batch_size, -1).norm(dim=1, keepdim=True)
            diff_norm = diff_flat.norm(dim=1, keepdim=True)
            scale = self.spherical_step * diff_norm.view(batch_size, 1, 1, 1) / (proj_norm.view(batch_size, 1, 1, 1) + 1e-12)
            
            # Apply the sideways wiggle
            candidate = candidate + scale * proj_noise
            
            # Clamp to be a valid image
            candidate = torch.clamp(candidate, 0, 1)
            
            # 3. The "Decision" Step
            # If this new candidate is STILL adversarial (The model is still fooled), we keep it!
            # If the model suddenly recognizes the dog errors, we reject this step and try again.
            is_adv = self._is_adversarial(candidate, labels)
            
            mask = is_adv
            if mask.any():
                curr_adv[mask] = candidate[mask]
                
            # Log progress every 500 steps
            if step % 500 == 0:
                # Calculate how close we are to the original (L2 distance)
                # Smaller is better!
                dist = (curr_adv - images).view(batch_size, -1).norm(dim=1).mean()

        return curr_adv
