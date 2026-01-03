# Hybrid Adversarial Defense System (Advanced)

## Project Overview
We developed a complete end-to-end framework to study and defend against adversarial attacks on image classifiers. Using **Tiny-ImageNet** as our testbed, we trained robust models, generated state-of-the-art attacks, and built a multi-layered detection system.

## What We Did

### 1. Robust Model Training
We successfully trained a custom **ResNet-18** model on the Tiny-ImageNet dataset (200 classes). By implementing strict validation checkpoints and learning rate scheduling, we achieved a stable baseline for our adversarial experiments.

### 2. Advanced Attack Generation
To rigorously test our defenses, we implemented **AutoAttack (Lite)**. This is an ensemble attack that combines:
- **PGD (Projected Gradient Descent)** with Cross-Entropy Loss.
- **APGD (Auto-PGD)** with Difference-of-Logits-Ratio (DLR) Loss.
This ensures we are testing against the "worst-case" perturbations, not just weak attacks.

### 3. Novel Detection Strategies
We explored three distinct approaches to detecting adversarial images:

*   **Prediction Stability (The "Physics" Defense)**:
    *   *Idea*: Real images are robust; adversarial noise is fragile.
    *   *Method*: We apply "aggressive" transformations (Resize to 28x28 + JPEG Compression) to the input.
    *   *Result*: This was our best performer, achieving an **ROC-AUC of 0.81**. It effectively "shatters" adversarial perturbations while keeping real images intact.

*   **Mahalanobis Distance (The "Statistical" Defense)**:
    *   *Idea*: Adversarial examples lie in low-probability regions of the feature space.
    *   *Method*: We modeled the distribution of features in the deep layers of the network.
    *   *Result*: Provided a theoretical baseline (AUC 0.53) but struggled with the high dimensionality of Tiny-ImageNet.

*   **Activation Detector (Experimental)**:
    *   *Idea*: A simple classifier should be able to separate clean vs. adversarial activations.
    *   *Method*: We trained a Logistic Regression on the penultimate layer features.
    *   *Result*: Served as a lightweight, learned baseline for comparison.
