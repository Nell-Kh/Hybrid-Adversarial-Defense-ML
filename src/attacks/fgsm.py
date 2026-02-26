import torch
import torch.nn as nn
from src.attacks.base import Attacker

class FGSMAttacker(Attacker):
    """Fast Gradient Sign Method (FGSM)"""
    def __init__(self, model, device, eps=0.03):
        super().__init__(model, device)
        self.eps = eps
        self.loss_fn = nn.CrossEntropyLoss()

    def attack(self, image, label):
        image = image.clone().detach().to(self.device)
        label = label.to(self.device).view(-1)
        
        image.requires_grad = True
        
        output = self.model(image)
        loss = self.loss_fn(output, label)
        
        self.model.zero_grad()
        loss.backward()
        
        data_grad = image.grad.data
        sign_data_grad = data_grad.sign()
        
        perturbed_image = image + self.eps * sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, -1, 1)
        
        return perturbed_image
