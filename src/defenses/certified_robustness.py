"""
[FILE PURPOSE]
Calculates the Certified Robustness Radius (R) using Randomized Smoothing.

[THE GOAL]
Empirical robustness (testing against specific attacks like DeepFool) is flawed 
because a better attack might be invented tomorrow that breaks the model.
Certified Robustness provides a MATHEMATICAL GUARANTEE. If an image is certified 
robust up to radius R, NO ATTACK in the universe with an L2 norm less than R 
can ever trick the model.

[HOW IT WORKS]
1. We inject Gaussian noise into the image N times.
2. We pass all N noisy copies through the base model and count the votes (like TTA).
3. We identify the majority class (c_A) and its probability (p_A).
4. Using the Neyman-Pearson lemma and the inverse CDF of the Gaussian distribution, 
   we calculate the absolute maximum distance an attacker could move the image 
   before the "vote mass" of c_A drops below 50%.
5. That distance is the Certified Radius (R).
"""

import torch
import math
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

class CertifiedRobustness:
    def __init__(self, model, device, noise_std=0.1, n0=20, n=200, alpha=0.001):
        """
        model: The base classifier (does NOT need to be robust itself, the smoothing creates the robustness)
        noise_std: Standard deviation of the Gaussian noise used for smoothing
        n0: Number of samples used to GUESS the top class
        n: Number of samples used to CERTIFY the top class
        alpha: Confidence level (e.g., 0.001 means we are 99.9% confident in the radius)
        """
        self.model = model
        self.device = device
        self.noise_std = noise_std
        self.n0 = n0
        self.n = n
        self.alpha = alpha

    def _sample_under_noise(self, x, num_samples):
        # Repeat the image num_samples times
        x_repeated = x.repeat(num_samples, 1, 1, 1)
        # Add pure Gaussian noise
        noise = torch.randn_like(x_repeated) * self.noise_std
        x_noisy = x_repeated + noise
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x_noisy)
            predictions = outputs.argmax(dim=1)
            
        return predictions

    def certify(self, x):
        """
        Returns:
        Prediction (int), Certified Radius (float)
        If the radius is 0.0, the prediction is NOT certified.
        """
        # 1. Guess the most likely class (c_hat) using a small sample (n0)
        counts0 = self._sample_under_noise(x, self.n0).bincount(minlength=200)
        c_hat = counts0.argmax().item()
        
        # 2. Evaluate the probability of c_hat using a large sample (n)
        counts = self._sample_under_noise(x, self.n).bincount(minlength=200)
        n_A = counts[c_hat].item()
        
        # 3. Calculate the lower bound of the probability of c_hat (pA_lower)
        # We use a Binomial confidence interval because we only sampled 'n' times, 
        # we didn't calculate exactly infinity times. This ensures rigorous safety.
        pA_lower, _ = proportion_confint(n_A, self.n, alpha=2 * self.alpha, method="beta")
        
        # 4. Calculate the Certified Radius (R)
        if pA_lower > 0.5:
            # Neyman-Pearson lemma application
            radius = self.noise_std * norm.ppf(pA_lower)
            return c_hat, radius, pA_lower
        else:
            # Model is too uncertain to certify anything
            return c_hat, 0.0, pA_lower
