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

# --- MAHALANOBIS DETECTOR ---
# SOTA unsupervised detection.
# We model the feature representations of each class as a Gaussian distribution.
# Adversarial examples (and OOD data) usually have high Mahalanobis distance 
# from the class distributions.

class MahalanobisDetector:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.class_means = {}
        self.precisionor = None # Inverse of covariance
        
    def fit(self, train_loader):
        print("Fitting Mahalanobis Detector (Calculating Statistics)...")
        # 1. Extract Features per class
        features_by_class = {}
        all_features = []
        
        # Hook for features
        curr_feats = {}
        def hook(m, i, o):
            curr_feats['out'] = o.flatten(1)
        handle = self.model.avgpool.register_forward_hook(hook)
        
        # Pass a subset of training data (100 batches is enough for good estimate)
        MAX_BATCHES = 100
        count = 0
        
        with torch.no_grad():
            for images, labels in tqdm(train_loader, desc="Extracting Train Features"):
                if count >= MAX_BATCHES: break
                images = images.to(self.device)
                _ = self.model(images)
                
                feats = curr_feats['out'].cpu().numpy()
                labels = labels.numpy()
                
                for f, l in zip(feats, labels):
                    if l not in features_by_class: features_by_class[l] = []
                    features_by_class[l].append(f)
                    all_features.append(f)
                    
                count += 1
                
        handle.remove()
        
        # 2. Compute Mean per class
        print("Computing Class Means...")
        for c, feats in features_by_class.items():
            self.class_means[c] = np.mean(feats, axis=0)
            
        # 3. Compute Shared Covariance (Empirical Covariance)
        # We assume tied covariance for stability (traditional Mahalanobis setup)
        print("Computing Precision Matrix...")
        X = np.array(all_features)
        # Center the data by class mean
        X_centered = []
        for i, l in enumerate(labels): # Note: this loop variable l is from the last batch, this is BUG.
             # Fixing logic: we need to subtract the specific class mean for each sample
             # But simpler way: Scikit-learn EmpiricalCovariance fits on the data.
             # Ideally we fit on (X - mean_class).
             pass

        # Correct way to compute shared covariance:
        # Subtract class mean from each sample
        X_centered = []
        # We need to re-iterate or store labels properly.
        # Let's just create a list of (feat, label) to be safe above.
        # Refactoring storage loop above is expensive.
        # Approximation: Fit global covariance or per-class? 
        # Paper says: "tied covariance": sum of (x - mu_c)(x - mu_c)^T
        
        # Let's do it properly. Re-looping over the dict we stored.
        for c, feats in features_by_class.items():
            mean = self.class_means[c]
            for f in feats:
                X_centered.append(f - mean)
                
        X_centered = np.array(X_centered)
        
        # Calculate covariance and precision (inverse)
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(X_centered)
        self.precisionor = ec.precision_
        print("Detector Fitted.")
        
    def score(self, images):
        # Calculate min Mahalanobis distance to any class
        
        # Extract features
        curr_feats = {}
        def hook(m, i, o):
            curr_feats['out'] = o.flatten(1)
        handle = self.model.avgpool.register_forward_hook(hook)
        
        with torch.no_grad():
            _ = self.model(images)
        feats = curr_feats['out'].cpu().numpy()
        handle.remove()
        
        scores = []
        for f in feats:
            # Distance to the CLOSEST class
            # Mahalanobis dist = (x - mu)^T * Sigma^-1 * (x - mu)
            # We compute this for all classes and take min? 
            # Actually, standard method is: take distance to the PREDICTED class or CLOSEST class.
            # We use closest class.
            
            min_dist = float('inf')
            
            # Optimization: We don't need to check 200 classes for every image if we trust the prediction?
            # But adversarial might change prediction.
            # Let's check all 200 (vectorized would be faster but loop is safer for now)
            
            for c, mean in self.class_means.items():
                diff = f - mean
                # dist = diff.T * Precision * diff
                dist = np.dot(np.dot(diff, self.precisionor), diff)
                if dist < min_dist:
                    min_dist = dist
            
            scores.append(min_dist)
            
        return np.array(scores)

def evaluate_mahalanobis():
    device = config.DEVICE
    print(f"Running Mahalanobis Evaluation on {device}...")
    
    # 1. Load Model
    model = get_model(device)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    model.eval()
    
    # 2. Fit Detector
    train_loader, val_loader = get_dataloaders()
    detector = MahalanobisDetector(model, device)
    detector.fit(train_loader)
    
    # 3. Evaluate
    print("Evaluating Detector on Clean vs AutoAttack...")
    
    attacker = AutoAttackLite(model, device)
    
    y_true = [] # 0 (clean), 1 (adv)
    y_scores = [] # Mahalanobis scores
    
    count = 0 
    MAX_TEST = 50 # Evaluate on 50 images
    
    for images, labels in tqdm(val_loader, desc="Testing"):
        if count >= MAX_TEST: break
        
        images, labels = images.to(device), labels.to(device)
        
        # A. Clean
        scores_clean = detector.score(images)
        y_true.extend([0] * len(images))
        y_scores.extend(scores_clean)
        
        # B. Attack
        # Filter for correct only (to be fair)
        with torch.no_grad(): preds = model(images).argmax(1)
        mask = preds == labels
        if not mask.any(): continue
        
        img_atk = images[mask]
        lbl_atk = labels[mask]
        
        adv_images = attacker.attack(img_atk, lbl_atk)
        scores_adv = detector.score(adv_images)
        
        y_true.extend([1] * len(adv_images))
        y_scores.extend(scores_adv)
        
        count += len(images)

    # AUC
    auc = roc_auc_score(y_true, y_scores)
    print(f"\n\n--- RESULTS (Mahalanobis SOTA) ---")
    print(f"ROC-AUC Score: {auc:.4f}")
    if auc > 0.85:
        print("Verdict: EXCELLENT. Use this in your report.")
    else:
        print("Verdict: Good, but training might need more epochs for tighter clusters.")

if __name__ == "__main__":
    evaluate_mahalanobis()
