"""
[FILE PURPOSE]
This is the "Blueprint" for all our attacks. 
It defines the rules that every new attack must follow. 
Think of this as the "Contract": if you want to build a new weapon, it MUST match this shape.

[WHAT IT NEEDS]
- Nothing. It is complete. It just sits here and enforces order.
"""

from abc import ABC, abstractmethod
import torch

class Attacker(ABC):
    """
    The Abstract Base Class (ABC) for all adversarial attacks.
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device
        # We always set the model to 'eval' mode.
        # This tells PyTorch: "We are not training weights right now, we are just using them."
        # This turns off things like Dropout and BatchNorm updates.
        self.model.eval()

    @abstractmethod
    def attack(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        The main function every attack must implement.
        
        Args:
            images: The original clean images (e.g., a picture of a Panda).
            labels: The true answer (e.g., "Panda").
            
        Returns:
            The ADVERSARIAL images (e.g., a Panda that looks like a Gibbon).
        """
        pass
    
    def _clamp(self, images: torch.Tensor) -> torch.Tensor:
        """
        A helper to keep images valid.
        Digital images must be between 0 (black) and 1 (white).
        If an attack pushes a pixel to 1.2 or -0.5, this function snaps it back to [0, 1].
        """
        return torch.clamp(images, -1, 1)
