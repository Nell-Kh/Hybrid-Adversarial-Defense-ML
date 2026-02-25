import torch
import torch.nn as nn
from torchvision import models
import config

def get_model(device=config.DEVICE, arch='resnet18'):
    """
    Returns a model modified for 200 classes.
    Args:
        device: torch device
        arch: 'resnet18', 'resnet50', or 'wide_resnet50_2'
    """
    print(f"Initializing {arch}...")
    
    if arch == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = model.fc.in_features
    elif arch == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_features = model.fc.in_features
    elif arch == 'wide_resnet50_2':
        model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.DEFAULT)
        num_features = model.fc.in_features
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    # Modify the final classification layer for Tiny-ImageNet (200 classes)
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