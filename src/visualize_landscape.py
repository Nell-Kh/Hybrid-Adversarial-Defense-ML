import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go
import sys
import os

# Adjust imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import config
from model import get_model
from dataset import get_dataloaders

def get_random_direction(model_input):
    """Generate a random normalized direction vector."""
    d = torch.randn_like(model_input)
    d = d / torch.norm(d.view(d.size(0), -1), dim=1, keepdim=True)
    return d

def visualize_loss_landscape(model, device, image, label, epsilon=8/255, steps=50):
    """
    Visualizes the loss landscape around a specific image.
    We move in two random directions (x and y axis on plot) and measure loss (z axis).
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    # 1. Define two random directions (basis vectors) for the plot
    d1 = get_random_direction(image)
    d2 = get_random_direction(image)
    
    # Grid range (from -epsilon to +epsilon)
    range_limit = epsilon * 2.0 
    x = np.linspace(-range_limit, range_limit, steps)
    y = np.linspace(-range_limit, range_limit, steps)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    print("Calculating loss landscape... this might take a minute...")
    
    for i in range(steps):
        for j in range(steps):
            # Perturb image: image + x_coord * d1 + y_coord * d2
            diff = torch.tensor(X[i, j]) * d1 + torch.tensor(Y[i, j]) * d2
            perturbed_image = image + diff.to(device)
            perturbed_image = torch.clamp(perturbed_image, -1, 1) # Clip to valid range
            
            with torch.no_grad():
                output = model(perturbed_image)
                loss = criterion(output, label)
                Z[i, j] = loss.item()
                
    # Plotly 3D Surface
    print("Generating Interactive Plotly Surface...")
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
    
    fig.update_layout(
        title=f'Deep Neural Loss Landscape (Epsilon={epsilon:.3f})',
        autosize=True,
        width=800,
        height=800,
        scene=dict(
            xaxis_title='Perturbation Direction 1',
            yaxis_title='Perturbation Direction 2',
            zaxis_title='Cross-Entropy Loss'
        ),
        margin=dict(l=65, r=50, b=65, t=90)
    )
    
    return fig

if __name__ == "__main__":
    device = config.DEVICE
    
    # Load Model
    model = get_model(device)
    path = os.path.join(config.MODEL_DIR, "resnet_robust.pth")
    
    # Handle checkpoint format (the fix we just did for other scripts)
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    # Get one image
    _, val_loader = get_dataloaders(batch_size=1)
    images, labels = next(iter(val_loader))
    images, labels = images.to(device), labels.to(device)
    
    visualize_loss_landscape(model, device, images, labels)
