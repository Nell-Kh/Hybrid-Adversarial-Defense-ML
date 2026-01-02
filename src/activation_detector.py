import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from model import get_model
from dataset import get_dataloaders
from attack import pgd_attack
from tqdm import tqdm

def extract_features(model, loader, device, max_batches=10):
    """
    Extracts internal features (from the layer before the final classification)
    for both Clean and Adversarial images.
    """
    features = []
    labels = []  # 0 for Clean, 1 for Adversarial
    
    # Hook to capture the feature vector (output of avgpool)
    extracted_features = {}
    def hook_fn(m, i, o):
        extracted_features['out'] = o.flatten(1)
        
    # Register hook on the average pooling layer (before fc)
    handle = model.avgpool.register_forward_hook(hook_fn)
    
    print(f"Extracting features from {max_batches} batches...")
    
    count = 0
    for images, _ in tqdm(loader):
        if count >= max_batches: break
        images = images.to(device)
        batch_size = images.size(0)
        
        # 1. Pass Clean Images
        _ = model(images)
        clean_feats = extracted_features['out'].detach().cpu().numpy()
        features.append(clean_feats)
        labels.append(np.zeros(batch_size))
        
        # 2. Generate & Pass Adversarial Images
        # Note: We use a dummy label here just to generate the noise
        dummy_labels = torch.zeros(batch_size, dtype=torch.long).to(device) 
        adv_images = pgd_attack(model, images, dummy_labels, device, steps=5) # Fewer steps for speed
        
        _ = model(adv_images)
        adv_feats = extracted_features['out'].detach().cpu().numpy()
        features.append(adv_feats)
        labels.append(np.ones(batch_size))
        
        count += 1
        
    handle.remove()
    
    X = np.concatenate(features)
    y = np.concatenate(labels)
    return X, y

def train_activation_detector():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running Activation Detector on {device}...")

    # Load Model
    model = get_model(device)
    try:
        model.load_state_dict(torch.load("../models/resnet_tinyimagenet.pth", map_location=device))
    except:
        print("Model not found! Please finish train.py first.")
        return
        
    model.eval()
    
    # Get Data
    loader = get_dataloaders()
    
    # Extract
    print("Generating dataset for detector...")
    X, y = extract_features(model, loader, device)
    
    # Train a simple classifier (Logistic Regression)
    # This learns the "signature" of adversarial attacks in the feature space
    print("Training Detector (Logistic Regression)...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    
    # Evaluate
    preds = clf.predict(X)
    probs = clf.predict_proba(X)[:, 1]
    
    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, probs)
    
    print(f"\n--- Detector 2 Results ---")
    print(f"Detector Accuracy: {acc*100:.2f}%")
    print(f"Detector ROC-AUC:  {auc:.4f}")
    print(f"Interpretation: The detector can distinguish Clean vs Fake features with {acc*100:.0f}% accuracy.")

if __name__ == "__main__":
    train_activation_detector()