# Hybrid Adversarial Defense System (Advanced)

A state-of-the-art framework for detecting and defending against adversarial attacks on image classifiers using Tiny-ImageNet. This project goes beyond standard methods, implementing ensemble attacks and statistical feature analysis.

## Core Components

### 1. Robust Target Model
- **ResNet-18** customized for Tiny-ImageNet (200 classes).
- Trained with strict validation checkpoints and learning rate scheduling.
- **Source**: `src/train.py`, `src/model.py`

### 2. Advanced Adversarial Attacks
We implement **State-of-the-Art** attack generation to rigorously test our defenses.
- **AutoAttack (Lite)**: An ensemble of PGD with Cross-Entropy Loss and Difference-of-Logits-Ratio (DLR) Loss. This ensures we find the worst-case perturbation.
- **PGD (Standard)**: Classic Projected Gradient Descent.
- **Source**: `src/auto_attack.py`, `src/attack.py`

### 3. Statistical Defense (Mahalanobis)
Instead of simple classifiers, we use **Mahalanobis Distance** to detect adversarial examples.
- **Mechanism**: We model the feature activations of every class as a Gaussian distribution.
- **Detection**: Adversarial examples typically lie in low-probability regions (far from the class mean). We calculate the Mahalanobis distance in the feature space to flag anomalies.
- **Source**: `src/mahalanobis_detector.py`

## How to Run

1.  **Install Dependencies**:
    ```bash
    pip install torch torchvision tqdm scikit-learn numpy
    ```

2.  **Evaluate Base Model**:
    ```bash
    python src/evaluate.py
    ```

3.  **Run Full Defense Evaluation**:
    This script generates AutoAttacks and tests the Mahalanobis detector against them.
    ```bash
    python src/evaluate_defense.py
    ```

## Project Summary & Achievements

- **Code Architecture**: Centralized `config.py` for reproducible research.
- **Data Integrity**: Validated data loading pipeline for Tiny-ImageNet.
- **Advanced Warfare**: Upgraded from simple PGD to AutoAttack strategies.
- **SOTA Detection**: Implemented statistical feature analysis (Mahalanobis) instead of basic supervised detection.

---
*Created by [Your Name]*
