import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        # Forward pass
        self.model.eval()
        # We need gradients even in eval mode for Grad-CAM
        # But usually model.eval() turns off dropout/batchnorm updates, 
        # we still need to set requires_grad=True for input if we were doing input grad,
        # but here we need weights to have grad? No, just acts.
        # Actually, we need to zero_grad and run backward.
        
        # Temporarily enable grad for the model parameters? 
        # No, we just need the graph to allow backward from output to layer.
        
        logits = self.model(x.detach())
        
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
            
        score = logits[0, class_idx]
        
        self.model.zero_grad()
        score.backward()
        
        # Generate heatmap
        gradients = self.gradients
        activations = self.activations
        
        # Global Average Pooling on gradients (Importance weights)
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activations
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        # Resize to input size
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return cam.detach().cpu().numpy()[0, 0]

def apply_heatmap(image_tensor, heatmap):
    """
    Overlays heatmap on image.
    image_tensor: (1, 3, H, W) normalized
    heatmap: (H, W) numpy [0, 1]
    """
    # Denormalize image for visualization
    # Assuming standard ImageNet mean/std or similar, but for vis we often just clamp 0-1
    # TinyImageNet in this repo seems to be 0-1 or normalized? 
    # Let's assume input is 0-1 tensor for app.py
    
    img = image_tensor.detach().squeeze().cpu().permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Superimpose
    overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
    return overlay
