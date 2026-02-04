import torch
import torch.nn as nn
from torchvision import models
import config

def get_model(device=config.DEVICE):
    """
    Returns a ResNet-18 model modified for 200 classes.
    """
    # Load pre-trained ResNet18
    # CONCEPT: Transfer Learning
    # We load a model that has already "read the internet" (ImageNet trained).
    # This gives us a huge head start compared to random weights.
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Modify the final classification layer for Tiny-ImageNet (200 classes)
    num_features = model.fc.in_features
    # CONCEPT: "Model Surgery"
    # The original ResNet has 1000 outputs. We cut that off and attach a new "head"
    # that has exactly 200 outputs for our specific dataset.
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)

    model = model.to(device)
    return model

if __name__ == "__main__":
    # Simple test to verify model architecture
    print(f"Testing model on {config.DEVICE}")
    net = get_model()
    test_input = torch.randn(1, 3, 64, 64).to(config.DEVICE)
    output = net(test_input)
    print(f"Model output shape: {output.shape}")