
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import mahalanobis

import sys
import os
import pickle

# Adjust imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
# Add src directory needed for 'import config' inside model.py
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import src.config as config
from src.model import get_model
from src.dataset import get_dataloaders

class MahalanobisDetector:
    """
    The 'Iron Dome': Multi-Layer Mahalanobis Detector (NeurIPS 2018).
    Checks anomalies at 5 different layers of ResNet.
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
        
        # Layers to hook
        # ResNet18 structure: conv1 -> layer1 -> layer2 -> layer3 -> layer4 -> avgpool
        self.layer_names = ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool']
        self.num_classes = config.NUM_CLASSES
        
        # Storage for statistics (per layer)
        # Structure: {'layer1': {'means': [200, dim], 'cov_inv': [dim, dim]}, ...}
        self.stats = {}
        
        # Meta-Classifier (Fuse 5 scores -> 1 probability)
        self.classifier = LogisticRegression(class_weight='balanced')
        self.trained = False

    def get_features(self, images, keep_graph=False):
        """
        Extract features from ALL monitored layers.
        Returns: Dictionary {layer_name: tensor_batch}
        """
        features = {name: [] for name in self.layer_names}
        
        def get_hook(name):
            def hook(model, input, output):
                # CONCEPT: Feature Extraction Hooks
                # Standard models are "black boxes" (Input -> Output).
                # We attach a "wire" (hook) to the internal layers to listen to their activations.
                # Global Average Pooling for Convolutional Layers to get [Batch, Channels]
                if output.dim() == 4:
                    # [B, C, H, W] -> [B, C]
                    out = torch.mean(output, dim=[2, 3])
                else:
                    out = output.flatten(1)
                
                if keep_graph:
                    features[name].append(out) # Keep on GPU, keep grad
                else:
                    features[name].append(out.detach().cpu())
            return hook
            
        # Register hooks
        hooks = []
        for name in self.layer_names:
            layer = getattr(self.model, name)
            hooks.append(layer.register_forward_hook(get_hook(name)))
            
        # Pass data
        with torch.no_grad():
            self.model(images.to(self.device))
            
        # Cleanup
        for h in hooks: h.remove()
        
        # Concatenate batches if needed (though we usually call this with one batch)
        # Modify to return the first element since the hook appends to a list
        final_features = {}
        for name in self.layer_names:
            final_features[name] = features[name][0]
            
        return final_features

    def fit_statistics(self, train_loader):
        """
        Step 1: Compute Class Means & Covariance for EACH layer.
        """
        print("--- PHASE 1: Calculating Layer-Wise Statistics ---")
        
        # Initialize storage
        layer_features = {name: [] for name in self.layer_names}
        all_labels = []
        
        # 1. Collect all features from training data
        for images, labels in tqdm(train_loader, desc="Scanning Dataset"):
            feats = self.get_features(images)
            for name in self.layer_names:
                layer_features[name].append(feats[name])
            all_labels.append(labels)
            
        # Concatenate
        all_labels = torch.cat(all_labels).numpy()
        for name in self.layer_names:
            layer_features[name] = torch.cat(layer_features[name]).numpy()
            
        # 2. Compute Stats per Layer
        for name in self.layer_names:
            print(f"Propagating {name}...")
            feats = layer_features[name]
            dim = feats.shape[1]
            
            # Dictionary for this layer
            self.stats[name] = {
                'means': np.zeros((self.num_classes, dim)),
                'cov_inv': None
            }
            
            shared_cov = np.zeros((dim, dim))
            
            for c in range(self.num_classes):
                class_feats = feats[all_labels == c]
                if len(class_feats) == 0: continue
                
                mean = np.mean(class_feats, axis=0)
                self.stats[name]['means'][c] = mean
                
                centered = class_feats - mean
                shared_cov += np.dot(centered.T, centered)
                
            shared_cov /= len(feats)
            
            # Invert with regularization
            self.stats[name]['cov_inv'] = np.linalg.inv(shared_cov + np.eye(dim) * 1e-5)
            
        print("Statistics Calculated.")
        self.save_stats()

    def get_mahalanobis_scores(self, images):
        """
        Compute the 5-dimensional score vector for a batch of images.
        Vector = [Score_Layer1, Score_Layer2, ..., Score_AvgPool]
        """
        batch_feats = self.get_features(images)
        batch_size = images.size(0)
        
        # Output: [Batch, 5]
        scores = np.zeros((batch_size, len(self.layer_names)))
        
        for i, name in enumerate(self.layer_names):
            feats = batch_feats[name].numpy()
            layer_stats = self.stats[name]
            
            for b in range(batch_size):
                feat = feats[b]
                
                # Find distance to NEAREST class
                min_dist = float('inf')
                for c in range(self.num_classes):
                    mean = layer_stats['means'][c]
                    # Optimized Mahalanobis: (x-u)T E^-1 (x-u)
                    delta = feat - mean
                    dist = np.dot(np.dot(delta, layer_stats['cov_inv']), delta)
                    if dist < min_dist:
                        min_dist = dist
                
                scores[b, i] = min_dist
                
        return scores

    def calc_torch_score(self, images):
        """
        Differentiable score calculation for Adaptive Attacks.
        Calculates the sum of Mahalanobis distances across all layers using PyTorch.
        """
        batch_feats = self.get_features(images, keep_graph=True)
        total_score = 0
        
        for name in self.layer_names:
            feat = batch_feats[name] # [B, Dim]
            
            # Convert stats to device
            means = torch.from_numpy(self.stats[name]['means']).float().to(self.device)
            cov_inv = torch.from_numpy(self.stats[name]['cov_inv']).float().to(self.device)
            
            x_minus_u = feat.unsqueeze(1) - means.unsqueeze(0) # [B, C, D]
            
            # Left term: (x-u) @ E^-1
            left = torch.matmul(x_minus_u, cov_inv)
            
            # Dot product: sum(left * right, dim=-1)
            dists = torch.sum(left * x_minus_u, dim=2) # [B, C]
            
            # Take min distance (closest class)
            min_dists, _ = torch.min(dists, dim=1)
            
            total_score += min_dists.sum()
            
        return total_score

    def train_classifier(self, train_loader, attack_fn, max_batches=20):
        """
        Step 2: Train the Logistic Regression (Fusion Layer).
        We need a dataset of (Clean vs Adversarial) examples.
        """
        print("\n--- PHASE 2: Training Fusion Classifier (Logistic Regression) ---")
        X = [] # List of score vectors [Batch, 5]
        y = [] # List of labels (0=Clean, 1=Adv)
        
        batches = 0
        for images, labels in tqdm(train_loader, desc="Generating Adversarial Data"):
            if batches >= max_batches: break
            batches += 1
            
            images = images.to(self.device).detach()
            labels = labels.to(self.device).detach()
            
            # 1. Get Scores for CLEAN images
            clean_scores = self.get_mahalanobis_scores(images)
            X.append(clean_scores)
            y.append(np.zeros(len(images))) # Label 0
            
            # 2. Generate ADVERSARIAL images
            # We assume attack_fn takes (model, images, labels) -> adv_images
            self.model.eval()
            adv_images = attack_fn(self.model, images, labels)
            
            # 3. Get Scores for ADV images
            adv_scores = self.get_mahalanobis_scores(adv_images)
            X.append(adv_scores)
            y.append(np.ones(len(images))) # Label 1
            
        # Combine
        X = np.concatenate(X, axis=0) # [N, 5]
        y = np.concatenate(y, axis=0) # [N]
        
        print(f"Training Data: {X.shape[0]} samples (50% Clean, 50% Adv)")
        
        # Train Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit Regressor
        print("Fitting Logistic Regression...")
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        train_acc = self.classifier.score(X_train, y_train)
        test_acc = self.classifier.score(X_test, y_test)
        print(f"Classifier Accuracy -> Train: {train_acc*100:.2f}% | Test: {test_acc*100:.2f}%")
        
        self.save_stats()
        self.trained = True

    def save_stats(self):
        data = {
            'stats': self.stats,
            'classifier': self.classifier,
            'trained': self.trained
        }
        with open('models/mahalanobis_stats.pkl', 'wb') as f:
            pickle.dump(data, f)
            
    def load_stats(self):
        if os.path.exists('models/mahalanobis_stats.pkl'):
            print("Loading Mahalanobis stats...")
            with open('models/mahalanobis_stats.pkl', 'rb') as f:
                data = pickle.load(f)
                
            # Handle backward compatibility if we reload old pkl without classifier
            if 'stats' in data:
                self.stats = data['stats']
                self.classifier = data.get('classifier', self.classifier)
                self.trained = data.get('trained', False)
            else:
                self.stats = data # Old format
                
            return True
        return False

if __name__ == "__main__":
    # Test Run
    device = config.DEVICE
    model = get_model(device)
    # Load the standard model to calculate stats
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    
    detector = MahalanobisDetector(model, device)
    
    # Needs training data to fit statstics
    train_loader, _ = get_dataloaders(batch_size=128)
    
    # 1. Fit Layer Statistics
    if not detector.load_stats():
        detector.fit_statistics(train_loader)
    
    # 2. Debug: Check scores for a clean batch
    print("Testing Detection Vector on CLEAN images...")
    images, _ = next(iter(train_loader))
    scores = detector.get_mahalanobis_scores(images)
    
    print("\n--- Detection Vectors (First 3 Images) ---")
    print("Format: [Layer1, Layer2, Layer3, Layer4, AvgPool]")
    for i in range(3):
        print(f"Image {i}: {scores[i]}")
        
    print("\nSystem Ready. Phase 2 (Calibration) required to train Regression.")
