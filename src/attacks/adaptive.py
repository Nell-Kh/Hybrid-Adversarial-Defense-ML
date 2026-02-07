
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
# Adjust path to enable absolute imports if running as script
if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '../')) # Add src to path for 'import config'
    
from src.attacks.base import Attacker
from src.attacks.auto_attack import AutoAttackLite

class AdaptiveAttacker(Attacker):
    """
    The 'Ninja' Attack (Adaptive PGD).
    
    Standard PGD optimizes:
        Maximize Loss(Model(x), y)
        
    Adaptive PGD optimizes:
        Maximize Loss(Model(x), y) - Lambda * AnomalyScore(x)
        
    Goal: Trick the model AND stay hidden from the Mahalanobis Detector.
    """
    def __init__(self, model, device, detector, eps=8/255, alpha=2/255, steps=20, lambda_param=0.5):
        super().__init__(model, device)
        self.detector = detector
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.lambda_param = lambda_param # Importance of staying hidden
        
    def attack(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.to(self.device)
        
        # Random Start
        adv_images = images + torch.empty_like(images).uniform_(-self.eps, self.eps)
        adv_images = torch.clamp(adv_images, -1, 1).detach()
        
        loss_fn = nn.CrossEntropyLoss()
        
        # Since we use an optimizer, we need to manually handle the projection
        # Or just use the standard PGD loop with manual grad update
        
        for i in range(self.steps):
            adv_images.requires_grad = True
            
            # 1. Model Loss (Fool the classifier)
            outputs = self.model(adv_images)
            model_loss = loss_fn(outputs, labels)
            
            # 2. Detector Loss (Fool the Iron Dome)
            # We want to MINIMIZE the Mahalanobis distance (look normal)
            # So in our Maximization loop, we SUBTRACT this term.
            anomaly_score = self.detector.calc_torch_score(adv_images)
            
            # Combined Objective
            # We want to maximize Model Loss (Error)
            # We want to minimize Anomaly Score (Detection) -> Maximize (-Score)
            total_loss = model_loss - (self.lambda_param * anomaly_score)
            
            # Update
            grad = torch.autograd.grad(total_loss, adv_images, retain_graph=False, create_graph=False)[0]
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            
            # Project back to Epsilon Ball
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=-1, max=1).detach()
            
        return adv_images

if __name__ == "__main__":
    # Test Run
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
    import src.config as config
    from src.model import get_model
    from src.dataset import get_dataloaders
    from src.defenses.mahalanobis import MahalanobisDetector
    
    device = config.DEVICE
    model = get_model(device)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    
    # Load Detector
    detector = MahalanobisDetector(model, device)
    if not detector.load_stats():
        print("Error: Train detector first!")
        exit()
        
    # Load Data
    train_loader, _ = get_dataloaders(batch_size=4)
    images, labels = next(iter(train_loader))
    
    # Grid Search for the Ultimate Ninja Attack
    print("\n⚔️ STARTING NINJA GRID SEARCH ⚔️")
    lambdas = [0.0, 1.0, 10.0, 100.0, 500.0]
    
    results = []
    
    for lam in lambdas:
        print(f"\nEvaluating Lambda = {lam}...")
        attacker = AdaptiveAttacker(model, device, detector, lambda_param=lam, steps=30)
        
        # Run Attack
        adv_imgs = attacker.attack(images, labels)
        
        # Metrics
        score = detector.calc_torch_score(adv_imgs).item()
        acc = (model(adv_imgs).argmax(1) == labels.to(device)).float().mean().item()
        
        print(f"  -> Anomaly Score: {score:.2f}")
        print(f"  -> Model Accuracy: {acc*100:.1f}% (Target: 0%)")
        
        results.append((lam, score, acc))

    print("\n=== 🏆 LEADBOARD 🏆 ===")
    print(f"Clean Image Score: {detector.calc_torch_score(images.to(device)).item():.2f}")
    print(f"{'Lambda':<10} | {'Score':<15} | {'Success Rate'}")
    print("-" * 45)
    
    for lam, score, acc in results:
        success = (1.0 - acc) * 100
        print(f"{lam:<10} | {score:<15.2f} | {success:.1f}%")
        
    print("\nInterpretation:")
    print("- Low Score + High Success = TRUE NINJA (Broken Defense)")
    print("- High Score + High Success = BRUTE (Caught by Detector)")
    print("- Low Score + Low Success = FAILED (Too constrained)")
