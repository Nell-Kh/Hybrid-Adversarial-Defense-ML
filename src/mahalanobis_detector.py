import torch
import torch.nn as nn
import numpy as np
import config
from model import get_model
from dataset import get_dataloaders
from auto_attack import AutoAttackLite
from tqdm import tqdm
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score
import os

# --- MAHALANOBIS DETECTOR (SOTA) ---
# Refined Implementation with Input Pre-processing
# Reference: "A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks" (NeurIPS 2018)

class MahalanobisDetector:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.class_means = {}
        self.precision = None 
        self.num_classes = config.NUM_CLASSES
        
    def extract_features(self, images):
        # Helper to get features from avgpool
        curr_feats = {}
        def hook(m, i, o):
            curr_feats['out'] = o.flatten(1)
        handle = self.model.avgpool.register_forward_hook(hook)
        
        with torch.enable_grad(): # Needed for input pre-processing
             out = self.model(images)
             
        feats = curr_feats['out']
        handle.remove()
        return feats, out

    def fit(self, train_loader):
        print("Fitting Mahalanobis Detector (Calculating Statistics)...")
        features_by_class = {}
        
        # 1. Extract All Features (Limit for speed if needed)
        MAX_SAMPLES = 2000 # 200 classes * 10 images each
        count = 0
        
        # Enable full extraction
        all_feats = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(train_loader, desc="Extracting Train Features"):
                if count >= MAX_SAMPLES: break
                images = images.to(self.device)
                
                # Get features
                curr_feats = {}
                def hook(m, i, o): curr_feats['out'] = o.flatten(1)
                h = self.model.avgpool.register_forward_hook(hook)
                _ = self.model(images)
                h.remove()
                
                batch_feats = curr_feats['out'].cpu().numpy()
                batch_labels = labels.numpy()
                
                for f, l in zip(batch_feats, batch_labels):
                    if l not in features_by_class: features_by_class[l] = []
                    features_by_class[l].append(f)
                    all_feats.append(f)
                    all_labels.append(l)
                    count += 1
        
        # 2. Compute Class Means
        print("Computing Class Means...")
        for c in range(self.num_classes):
             if c in features_by_class:
                 self.class_means[c] = np.mean(features_by_class[c], axis=0)
             else:
                 # Fallback if class not sampled
                 self.class_means[c] = np.zeros_like(all_feats[0])

        # 3. Compute Tied Covariance
        print("Computing Precision Matrix...")
        X_centered = []
        for f, l in zip(all_feats, all_labels):
            class_mean = self.class_means[l]
            X_centered.append(f - class_mean)
            
        X_centered = np.array(X_centered)
        
        # Fit covariance
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(X_centered)
        
        # Store Precision Matrix (Inverse Covariance) as Tensor for fast compute
        self.precision = torch.from_numpy(ec.precision_).float().to(self.device)
        self.class_means_tensor = torch.from_numpy(np.array([self.class_means[c] for c in range(self.num_classes)])).float().to(self.device)
        
        print("Detector Fitted.")

    def score(self, images, noise_magnitude=0.002):
        # Implements Input Pre-processing:
        # We add small noise to x to DECREASE the Mahalanobis distance to the closest class.
        # This makes clean images "closer" and adversarial images "harder to fix", enhancing detection.
        
        images.requires_grad = True
        
        # 1. Get features of original image
        features, outcomes = self.extract_features(images)
        features = features.view(features.size(0), -1)
        
        # 2. Find closest class index (approximate by model prediction or closest mean)
        # Using model prediction is faster
        pred_labels = outcomes.argmax(1)
        
        # 3. Calculate Gradient w.r.t Input to Minimize Mahalanobis Distance
        # Distance = (f(x) - mu)^T * P * (f(x) - mu)
        
        # Gather means for the predicted classes
        means = self.class_means_tensor.index_select(0, pred_labels)
        diff = features - means
        
        # Term: P * (f(x) - mu)
        # P is (D, D), diff is (B, D) -> (B, D)
        term = torch.mm(diff, self.precision) 
        
        # Dist = batch dot product
        dist = (term * diff).sum(dim=1)
        
        # Compute gradients
        # We want to minimize distance -> move x against gradient of distance
        # But wait, the standard method adds gradients to MAXIMIZE score? 
        # Usually: x_new = x - epsilon * sign(grad(dist))
        
        grads = torch.autograd.grad(dist.sum(), images, retain_graph=False)[0]
        
        # Update Image (Input Pre-processing)
        images_new = images - noise_magnitude * grads.sign()
        images_new = torch.clamp(images_new, -1, 1).detach() # assuming normalized to -1..1 or similar? 
        # Our data is 0.5 mean, 0.5 std -> range -1 to 1.
        
        # 4. Re-compute Score on Pre-processed Image
        with torch.no_grad():
            features_new, _ = self.extract_features(images_new)
            features_new = features_new.view(features_new.size(0), -1)
            
        # Compute distance to ALL classes and take Min
        # Doing loop for safety with memory
        
        batch_scores = []
        for i in range(features_new.size(0)):
            f = features_new[i] # (D,)
            
            # Broadcast f to (C, D)
            f_expand = f.unsqueeze(0).expand(self.num_classes, -1)
            
            diff = f_expand - self.class_means_tensor
            # (C, D) x (D, D) -> (C, D)
            term = torch.mm(diff, self.precision)
            
            # (C, D) * (C, D) -> sum -> (C,)
            dists = (term * diff).sum(dim=1)
            
            # Score is the MIN distance (negative of "confidence")
            # Usually detection uses -min_dist (so higher is better/safer)
            # But for "Anomaly Score", Higher is Anomalous. So we return min_dist.
            batch_scores.append(dists.min().item())
            
        return np.array(batch_scores)

def evaluate_mahalanobis():
    device = config.DEVICE
    print(f"Running Mahalanobis Evaluation (SOTA Mode) on {device}...")
    
    # Load Model
    model = get_model(device)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    model.eval()
    
    # Fit Detector
    train_loader, val_loader = get_dataloaders()
    detector = MahalanobisDetector(model, device)
    detector.fit(train_loader)
    
    attacker = AutoAttackLite(model, device)
    
    y_true = [] 
    y_scores = [] 
    
    count = 0 
    MAX_TEST = 50 
    
    for images, labels in tqdm(val_loader, desc="Testing"):
        if count >= MAX_TEST: break
        
        images, labels = images.to(device), labels.to(device)
        
        # Clean
        scores_clean = detector.score(images)
        y_true.extend([0] * len(images))
        y_scores.extend(scores_clean)
        
        # Attack (Correctly filtered)
        with torch.no_grad(): preds = model(images).argmax(1)
        mask = preds == labels
        if not mask.any(): continue
        
        images = images[mask]
        labels = labels[mask]
        
        adv_images = attacker.attack(images, labels)
        scores_adv = detector.score(adv_images)
        
        y_true.extend([1] * len(adv_images))
        y_scores.extend(scores_adv)
        
        count += len(images)

    auc = roc_auc_score(y_true, y_scores)
    print(f"\n\n--- RESULTS (Mahalanobis SOTA) ---")
    print(f"ROC-AUC Score: {auc:.4f}")
    if auc > 0.6:
        print("Verdict: SUCCESS. The detector is working.")
    else:
        print("Verdict: Still struggling. The model's feature space might be too entangled.")

if __name__ == "__main__":
    evaluate_mahalanobis()
