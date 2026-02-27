"""
[FILE PURPOSE]
This file implements the "Expectation over Transformation" (EoT) adaptive attack.
Code name: The Oracle.
It is an ADVANCED WHITE-BOX attack explicitly designed to defeat randomized defenses.

[THE GOAL]
Our standard attacks failed against the Stochastic Ensemble (TTA) because the 
defense randomly shifted the image, breaking the static gradient maps.
The Oracle (EoT) assumes the attacker KNOWS we are using a randomized defense.

[HOW IT WORKS]
Instead of calculating the gradient on a static image, the Oracle simulates the 
defense's random shifts during optimization. It generates a batch of randomly 
transformed copies of the input, calculates the gradient for ALL of them, 
and takes the mathematical EXPECTATION (average) gradient. 

By following the average gradient of all possible spatial shifts, it crafts an 
adversarial pattern so deeply rooted that it survives random translations.
"""

from .base import Attacker
import torch
import torch.nn as nn
import torch.nn.functional as F

class EoTAttacker(Attacker):
    def __init__(self, model, device, eps=8/255, alpha=2/255, steps=20, eot_samples=10, max_shift=2):
        super().__init__(model, device)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        
        # EoT Specific Parameters
        self.eot_samples = eot_samples # How many parallel universes to simulate per step
        self.max_shift = max_shift     # Must match or exceed the defense's variance
        
    def get_eot_gradient(self, image, label, loss_fn):
        """
        Simulates the defense's randomized transformations to calculate the 
        expected (average) gradient over the distribution of possible shifts.
        """
        b, c, h, w = image.shape
        image.requires_grad = True
        
        # We accumulate gradients over EOT samples
        expected_grad = torch.zeros_like(image).to(self.device)
        
        padding = (self.max_shift, self.max_shift, self.max_shift, self.max_shift)
        
        for _ in range(self.eot_samples):
            # Apply transformation
            x_padded = F.pad(image, padding, mode='reflect')
            dx = torch.randint(0, self.max_shift * 2 + 1, (1,)).item()
            dy = torch.randint(0, self.max_shift * 2 + 1, (1,)).item()
            x_shifted = x_padded[:, :, dy:dy+h, dx:dx+w]
            
            # Forward pass
            outputs = self.model(x_shifted)
            loss = loss_fn(outputs, label)
            
            # Backward pass
            grad = torch.autograd.grad(loss, image, retain_graph=False, create_graph=False)[0]
            expected_grad += grad
            
        expected_grad = expected_grad / self.eot_samples
        return expected_grad

    def attack(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        adv_images = images.clone().detach()
        # Random uniform start (helps break ties and flat spots)
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
        adv_images = torch.clamp(adv_images, -1, 1).to(self.device).detach()
        
        loss_fn = nn.CrossEntropyLoss()
        
        batch_size = images.size(0)
        final_adv = torch.zeros_like(images)
        
        for b in range(batch_size):
            img = adv_images[b:b+1].detach()
            lbl = labels[b:b+1]
            orig_img = images[b:b+1]
            
            # Reset gradients for this image's loop
            
            for step in range(self.steps):
                expected_grad = self.get_eot_gradient(img, lbl, loss_fn)
                
                # Update the adversarial image using the Expected Gradient
                img = img.detach() + self.alpha * expected_grad.sign()
                
                # Project back into the Valid Epsilon hypersphere
                delta = torch.clamp(img - orig_img.to(self.device), min=-self.eps, max=self.eps)
                img = torch.clamp(orig_img.to(self.device) + delta, min=-1, max=1).detach()
                
            final_adv[b] = img[0].detach()
            
        return final_adv
