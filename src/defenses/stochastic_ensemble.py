import torch
import torch.nn as nn
import torch.nn.functional as F

class TTA_Ensemble(nn.Module):
    """
    Test-Time Augmentation (Stochastic Ensemble) Defense.
    
    Instead of passing the image through the model once, we generate N slightly 
    varied copies (using micro-translations and noise) on the fly. We pass all 
    N copies through the model and gather the "Majority Consensus Vote".
    
    Why it works: Adversarial attacks are brittle. They rely on finding highly 
    specific microscopic grid patterns. If we shift the image by even 1 pixel 
    or add random Gaussian noise, the attack's gradient path shatters, but the 
    underlying object (a dog) remains perfectly recognizable to the network.
    """
    def __init__(self, base_model, num_copies=10, max_shift=2, noise_std=0.02):
        super(TTA_Ensemble, self).__init__()
        self.base_model = base_model
        self.num_copies = num_copies
        self.max_shift = max_shift
        self.noise_std = noise_std
        
    def forward(self, x, return_consensus=False):
        """
        x: (1, C, H, W)
        Returns: 
        - If return_consensus=False: Just the averaged logit tensor (1, NumClasses)
        - If return_consensus=True: Average Logits AND the breakdown of votes
        """
        b, c, h, w = x.shape
        device = x.device
        
        # 1. Create a batch of N copies
        x_copies = x.repeat(self.num_copies, 1, 1, 1)
        
        # 2. Apply Micro-Translations (Pad and Crop)
        # We pad the image by max_shift, then randomly crop back to original size
        padding = (self.max_shift, self.max_shift, self.max_shift, self.max_shift)
        x_padded = F.pad(x_copies, padding, mode='reflect')
        
        x_shifted = torch.zeros_like(x_copies)
        for i in range(self.num_copies):
            # The first copy is always the original un-altered image
            if i == 0:
                x_shifted[i] = x[0]
                continue
                
            dx = torch.randint(0, self.max_shift * 2 + 1, (1,)).item()
            dy = torch.randint(0, self.max_shift * 2 + 1, (1,)).item()
            x_shifted[i] = x_padded[i, :, dy:dy+h, dx:dx+w]
            
        # 3. Inject Gaussian Noise (Shatter the adversarial gradient map)
        noise = torch.randn_like(x_shifted) * self.noise_std
        x_stochastic = torch.clamp(x_shifted + noise, -1.0, 1.0)
        
        # Ensure the first image remains perfectly clean just in case
        x_stochastic[0] = x[0]
        
        # 4. Forward Pass all N copies
        # Set to eval mode to ensure BatchNorm uses running stats, not batch stats
        self.base_model.eval() 
        with torch.no_grad():
            logits = self.base_model(x_stochastic)
            
        # 5. Average the logits (Soft majority voting)
        avg_logits = logits.mean(dim=0, keepdim=True)
        
        if not return_consensus:
            return avg_logits
            
        # Optional: Calculate hard majority voting breakdown for UI presentation
        predictions = logits.argmax(dim=1)
        
        # Count occurrences of each prediction
        vote_counts = {}
        for pred in predictions:
            p = pred.item()
            vote_counts[p] = vote_counts.get(p, 0) + 1
            
        # Sort by most votes
        sorted_votes = dict(sorted(vote_counts.items(), key=lambda item: item[1], reverse=True))
        
        return avg_logits, sorted_votes
