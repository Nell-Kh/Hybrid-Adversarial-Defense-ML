import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class ImageCleaner:
    """
    Base class for image cleaning defenses.
    """
    def __init__(self, device):
        self.device = device
        
    def clean(self, img_tensor):
        raise NotImplementedError

class GaussianDenoise(ImageCleaner):
    def __init__(self, device, kernel_size=3, sigma=1.0):
        super().__init__(device)
        self.kernel_size = kernel_size
        self.sigma = sigma
        
    def clean(self, img_tensor):
        # img_tensor shape: [B, C, H, W]
        # values in [-1, 1], we generally map them back to [-1, 1]
        
        # torchvision's gaussian_blur can take tensors
        cleaned = TF.gaussian_blur(img_tensor, kernel_size=[self.kernel_size, self.kernel_size], sigma=[self.sigma, self.sigma])
        return cleaned

class BitDepthReduction(ImageCleaner):
    def __init__(self, device, bits=3):
        super().__init__(device)
        self.bits = bits
        
    def clean(self, img_tensor):
        # Image is typically in range [-1, 1]. Shift to [0, 1]
        shifted = (img_tensor + 1.0) / 2.0
        
        # Quantize
        num_bins = 2 ** self.bits
        quantized = torch.round(shifted * (num_bins - 1)) / (num_bins - 1)
        
        # Shift back to [-1, 1]
        return quantized * 2.0 - 1.0

class MedianDenoise(ImageCleaner):
    def __init__(self, device, kernel_size=3):
        super().__init__(device)
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def clean(self, img_tensor):
        # Apply median filter using unfold
        B, C, H, W = img_tensor.shape
        # Pad image
        padded = F.pad(img_tensor, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
        # Extract patches: [B, C * K * K, H * W]
        patches = F.unfold(padded, kernel_size=self.kernel_size)
        # Reshape to [B, C, K*K, H*W]
        patches = patches.view(B, C, self.kernel_size * self.kernel_size, -1)
        # Compute median along the K*K dimension
        median, _ = patches.median(dim=2)
        # Reshape back to [B, C, H, W]
        cleaned = median.view(B, C, H, W)
        return cleaned

def apply_cleaning(img_tensor, method_name, device):
    """
    Factory function to apply the requested cleaning method.
    """
    if method_name == "Gaussian Blur":
        cleaner = GaussianDenoise(device, kernel_size=3, sigma=0.5)
    elif method_name == "Bit Depth Reduction (3-bit)":
        cleaner = BitDepthReduction(device, bits=3)
    elif method_name == "Bit Depth Reduction (4-bit)":
        cleaner = BitDepthReduction(device, bits=4)
    elif method_name == "Bit Depth Reduction (5-bit)":
        cleaner = BitDepthReduction(device, bits=5)
    elif method_name == "Bit Depth Reduction (6-bit)":
        cleaner = BitDepthReduction(device, bits=6)
    elif method_name == "Bit Depth Reduction (7-bit)":
        cleaner = BitDepthReduction(device, bits=7)
    elif method_name == "Median Filter":
        cleaner = MedianDenoise(device, kernel_size=3)
    else:
        return img_tensor # "None" or unknown
        
    return cleaner.clean(img_tensor)
