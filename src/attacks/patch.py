"""
[FILE PURPOSE]
This file implements the "Adversarial Patch" (The Sticker).
It is a PHYSICAL ATTACK.

[THE GOAL]
Most attacks change every pixel slightly (invisible noise).
This attack changes a small square COMPLETELY (visible noise).
The goal is to create a "Universal" pattern (like a QR code) that works on ANY image.

[HOW IT WORKS]
1. We have a small 24x24 pixel square (The Patch).
2. We place it on top of a random image (e.g., a Frog).
3. We optimize the pixels in the square to make the model scream "GOLDFISH".
4. EOT (Expectation Over Transformation): We randomly rotate, scale, and move the patch during training.
   This teaches the patch to survive in the real world (e.g., if you print it out and hold it at an angle).
"""

from .base import Attacker
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import config

class PatchAttacker(Attacker):
    """
    The main wrapper that lets us use the Patch like any other attack.
    """
    def __init__(self, model, device, patch_path=None):
        super().__init__(model, device)
        self.patch_path = patch_path
        self.patch = None
        # The tool that actually puts the patch on the image
        self.applier = PatchApplier(device)
        
        # Load the pre-trained sticker
        if patch_path and os.path.exists(patch_path):
            self.patch = torch.load(patch_path, map_location=device)

    def attack(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Applies the patch to the images.
        """
        if self.patch is None:
            # If we don't have a trained patch, we just use random noise.
            # (It won't work well, but it prevents crashing).
            self.patch = torch.rand(3, 24, 24).to(self.device)
            
        # Paste the sticker!
        return self.applier(images, self.patch)

class PatchApplier(nn.Module):
    """
    The geometric engine. It handles the "EOT" (random placement).
    """
    def __init__(self, device, img_size=64, min_scale=0.2, max_scale=0.5):
        super().__init__()
        self.device = device
        self.img_size = img_size
        self.min_scale = min_scale
        self.max_scale = max_scale

    def forward(self, img_batch, adv_patch):
        # We loop through every image in the batch and paste the patch in a different random spot.
        
        batch_size = img_batch.size(0)
        adv_batch = img_batch.clone()
        
        for i in range(batch_size):
            # 1. Random Size (Scale)
            # Simulated holding the sticker closer or further away.
            scale = random.uniform(self.min_scale, self.max_scale)
            target_h = int(self.img_size * scale)
            target_w = int(self.img_size * scale)
            
            # Resize patch to this new scale
            patch_resized = F.interpolate(
                adv_patch.unsqueeze(0), 
                size=(target_h, target_w), 
                mode='bilinear'
            ).squeeze(0)
            
            # 2. Random Location
            # Where is the sticker? Top-left? Center?
            max_x = self.img_size - target_w
            max_y = self.img_size - target_h
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            
            # 3. Paste it
            adv_batch[i, :, y:y+target_h, x:x+target_w] = patch_resized
            
        return adv_batch
