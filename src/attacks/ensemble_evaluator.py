import torch
import sys
import os
# Adjust path to enable absolute imports if running as script
if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from src.attacks.auto_attack import AutoAttackLite
from src.attacks.deepfool import DeepFool
from src.attacks.boundary import BoundaryAttack
from src.attacks.cw import CWAttacker
from tqdm import tqdm

class EnsembleEvaluator:
    """
    Standardizes the evaluation of robust models by running a suite
    of diverse attacks (Gradient, Optimization, Decision-based) and
    reporting the worst-case robust accuracy.
    """
    def __init__(self, model, device, eps=8/255, steps=50):
        self.model = model
        self.device = device
        self.eps = eps
        self.steps = steps
        
        # Initialize standard suite
        self.attackers = {
            "AutoAttack (Linf)": AutoAttackLite(model, device, eps=eps, steps=steps),
            "DeepFool (L2)": DeepFool(model, device, max_iters=steps),
            "C&W (L2)": CWAttacker(model, device, steps=steps, c=1.0)
        }

    def evaluate(self, dataloader, num_batches=1):
        """
        Runs the ensemble suite over the dataloader and calculates
        the worst-case accuracy (an image is robust ONLY if it survives ALL attacks).
        """
        self.model.eval()
        total = 0
        clean_correct = 0
        
        # Tracks how many images each attack successfully fooled
        attack_successes = {name: 0 for name in self.attackers}
        
        # Tracks worst-case robust correct (survived ALL attacks)
        robust_correct = 0

        print(f"Running Ensemble Evaluation on {self.device} for {num_batches} batches...")
        
        for i, (images, labels) in enumerate(tqdm(dataloader, desc="Evaluator Batches", total=num_batches)):
            if i >= num_batches: break
            
            images, labels = images.to(self.device), labels.to(self.device)
            batch_size = images.size(0)
            total += batch_size
            
            # --- 1. Clean Accuracy ---
            with torch.no_grad():
                clean_preds = self.model(images).argmax(1)
                clean_mask = (clean_preds == labels)
                clean_correct += clean_mask.sum().item()
            
            # If the model already got it wrong cleanly, no need to attack it for worst-case tracking
            # But we still evaluate to see attack specific success rates. 
            # We track a boolean tensor for whether each image "survived"
            survived_all = clean_mask.clone()
            
            # --- 2. Run Attacks ---
            for name, attacker in self.attackers.items():
                # Attack the batch
                adv_images = attacker.attack(images, labels)
                
                with torch.no_grad():
                    adv_preds = self.model(adv_images).argmax(1)
                    adv_mask = (adv_preds == labels)
                    
                    # Attack succeeded if the prediction is now wrong (and was originally right)
                    success_mask = clean_mask & ~adv_mask
                    attack_successes[name] += success_mask.sum().item()
                    
                    # Update survival mask (survives if it survived before AND survived this attack)
                    survived_all = survived_all & adv_mask
            
            robust_correct += survived_all.sum().item()
            
        results = {
            "Total Images": total,
            "Clean Accuracy (%)": (clean_correct / total) * 100,
            "Worst-Case Robust Accuracy (%)": (robust_correct / total) * 100,
        }
        
        for name in self.attackers:
            # Success Rate is out of ONLY the correctly classified images
            if clean_correct > 0:
                results[f"{name} Success Rate (%)"] = (attack_successes[name] / clean_correct) * 100
            else:
                results[f"{name} Success Rate (%)"] = 0.0
                
        return results

if __name__ == "__main__":
    from src.model import get_model
    from src.dataset import get_dataloaders
    import src.config as config
    import os
    
    device = config.DEVICE
    victim = get_model(device)
    victim_path = os.path.join(config.MODEL_DIR, "resnet_tinyimagenet.pth")
    victim.load_state_dict(torch.load(victim_path, map_location=device))
    
    _, val_loader = get_dataloaders(batch_size=16)
    
    evaluator = EnsembleEvaluator(victim, device, steps=20)
    res = evaluator.evaluate(val_loader, num_batches=2)
    
    print("\n=== Ensemble Evaluation Results ===")
    for k, v in res.items():
        if isinstance(v, float):
            print(f"{k}: {v:.1f}")
        else:
            print(f"{k}: {v}")
