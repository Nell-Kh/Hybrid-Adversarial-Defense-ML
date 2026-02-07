"""
[FILE PURPOSE]
This file implements "DeepFool" (The Minimalist).
It is a WHITE-BOX attack (needs gradients).

[THE GOAL]
Most attacks just want to break the model (find ANY error).
DeepFool is different. It wants to find the SMALLEST possible change to break the model.
It tries to answer: "What is the shortest path to the decision boundary?"

[HOW IT WORKS]
1. Imagine the "Decision Boundary" (the line between "Cat" and "Dog") is a flat wall.
2. We calculate the vector that points straight at that wall (using gradients).
3. We take a step in that direction.
4. Because the boundary isn't actually flat (it's curved), we might not cross it yet.
5. So we repeat: "Where is the wall now?" and step again.
6. We stop as soon as the label changes.

[WHAT IT NEEDS]
- It can be unstable if the gradient is zero (I added a check for this).
- It calculates the "L2 Norm" (Euclidean distance) to measure efficiency.
- **NEW**: If gradients are flat (the model is too confident), it adds random "Jitter" to escape the plateau.
"""

from .base import Attacker
import torch
import torch.nn.functional as F
import config

class DeepFool(Attacker):
    def __init__(self, model, device, max_iters=50, overshoot=0.02):
        super().__init__(model, device)
        self.max_iters = max_iters
        
        # A tiny extra push to make sure we actually cross the boundary line
        self.overshoot = overshoot

    def attack(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # We loop over images one by one because each image might need a different number of steps.
        # Some images are near the edge (easy to break), some are safe in the middle (hard).
        
        batch_size = images.size(0)
        input_shape = images.shape[1:]
        
        adv_images = images.clone()
        
        print("Running DeepFool (Finding shortest path to error)...")
        
        for b in range(batch_size):
            image = images[b:b+1].requires_grad_()
            original_label = labels[b].item()
            
            # The accumulated noise we added so far
            w = torch.zeros(input_shape).to(self.device)
            r_tot = torch.zeros(input_shape).to(self.device)
            
            # Start loop
            pert_image = image
            
            # Get model prediction
            outputs = self.model(pert_image)
            _, current_label = torch.max(outputs, 1)
            
            i = 0
            while current_label == original_label and i < self.max_iters:
                
                # 1. Get Gradient of the Original Class
                # "Which direction makes the model MORE confident it's a Cat?"
                outputs[0, original_label].backward(retain_graph=True)
                grad_orig = image.grad.data.clone()
                image.grad.zero_()
                
                # 2. Find the Nearest Other Class
                # We check every other class (Dog, Truck, Frog...) and see which one is "closest".
                # Closest means: (Difference in Score) / (Difference in Gradient) is minimal.
                
                start_grad = grad_orig
                min_pert = float('inf')
                best_w = None
                
                for k in range(outputs.size(1)):
                    if k == original_label: continue
                    
                    # Gradient for class k
                    image.grad.zero_()
                    outputs[0, k].backward(retain_graph=True)
                    grad_k = image.grad.data.clone()
                    
                    # The vector connecting the two gradients
                    w_k = grad_k - grad_orig
                    # The difference in confidence scores
                    f_k = outputs[0, k] - outputs[0, original_label]
                    
                    # Calculate distance
                    w_norm = w_k.norm()
                    pert_k = abs(f_k.item()) / (w_norm + 1e-8)
                    
                    # Keep the smallest one
                    if pert_k < min_pert and pert_k < 10.0:
                        min_pert = pert_k
                        best_w = w_k
                
                # 3. Accumulated the Perturbation
                if min_pert == float('inf'):
                    print(f"    Warning: Gradients are flat for label {original_label}. Adding STRONG random jitter.")
                    r_i = torch.randn_like(image).to(self.device) * 0.1
                elif best_w is not None:
                    # Direction = best_w / norm
                    # Magnitude = min_pert
                    best_w_norm = torch.norm(best_w)
                    r_i = (min_pert + 1e-4) * best_w / (best_w_norm + 1e-8)
                else:
                    r_i = torch.zeros_like(image).to(self.device)

                r_tot = r_tot + r_i
                
                # 4. Apply the change
                pert_image = image + (1 + self.overshoot * 10) * r_tot
                pert_image = torch.clamp(pert_image, -1, 1).detach().requires_grad_()
                
                # 5. Check prediction again
                outputs = self.model(pert_image)
                _, current_label = torch.max(outputs, 1)
                
                # [EXPLAINER LOGGING]
                if b == 0: # Only print for the first image to avoid spam
                    print(f"  Iteration {i}: Label {original_label} -> {current_label.item()}")
                    if best_w is not None:
                        print(f"    Nearest Class: {outputs.argmax(1).item()} | Distance to Boundary: {min_pert:.6f}")
                        print(f"    Step Size: {torch.norm(r_i):.6f}")
                
                if current_label != original_label:
                    # We crossed the boundary!
                    break

                i += 1
            
            adv_images[b] = pert_image.detach()
            
        return adv_images
