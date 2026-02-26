"""
[FILE PURPOSE]
This module defines the Deep Denoising Autoencoder (DAE).
It is a Convolutional Neural Network (CNN) specifically designed to "clean"
images that have been corrupted by adversarial noise.

[HOW IT WORKS]
1. ENCODER: Compresses the 64x64 image down into a dense 16x16 representation. 
   This forces the network to throw away fine-grained, unstructured data 
   (like adversarial static) and keep only the core semantic geometry.
2. DECODER: Reconstructs the 64x64 image from the latent space. Because the 
   adversarial noise was destroyed in the bottleneck, the Decoder reconstructs
   a perfectly pristine, clean image that the standard Baseline model can read.
"""
import torch
import torch.nn as nn
from .image_cleaning import ImageCleaner

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        
        # --- THE ENCODER (Compression) ---
        # Input: [B, 3, 64, 64] -> Output: [B, 64, 16, 16]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        
        # --- THE DECODER (Reconstruction) ---
        # Input: [B, 64, 16, 16] -> Output: [B, 3, 64, 64]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # Outputs values in range [-1, 1] to match standard normalized image tensors
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

class NeuralCleaner(ImageCleaner):
    """
    Wrapper class that conforms to our ImageCleaner interface in app.py
    """
    def __init__(self, device, weights_path=None):
        super().__init__(device)
        self.model = DenoisingAutoencoder().to(self.device)
        
        if weights_path is not None:
            try:
                self.model.load_state_dict(torch.load(weights_path, map_location=device))
                self.model.eval()
                print(f"[+] Loaded Neural Cleaner weights from {weights_path}")
            except FileNotFoundError:
                print(f"[-] Could not find weights at {weights_path}. Model will output random noise.")
                
    def clean(self, img_tensor):
        # We assume the image tensor is already bounded [-1, 1]
        self.model.eval()
        with torch.no_grad():
            cleaned_img = self.model(img_tensor)
        return cleaned_img
