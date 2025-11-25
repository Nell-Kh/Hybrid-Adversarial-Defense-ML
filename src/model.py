import torch
import torch.nn as nn
from torchvision import models

def get_model(device="cpu"):
    """
    Returns a ResNet-18 model modified for 200 classes.
    """
    # Load pre-trained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Modify the final classification layer for Tiny-ImageNet (200 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 200)

    model = model.to(device)
    return model

if __name__ == "__main__":
    # Simple test to verify model architecture
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    net = get_model(device)
    test_input = torch.randn(1, 3, 64, 64).to(device)
    output = net(test_input)
    print(f"Model output shape: {output.shape}")