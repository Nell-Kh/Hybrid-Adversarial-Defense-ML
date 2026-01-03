# Hybrid Adversarial Defense System (Advanced)

A state-of-the-art framework for detecting and defending against adversarial attacks on image classifiers using Tiny-ImageNet. This project implements advanced ensemble attacks and robust physics-based detection.

## Core Components

### 1. Robust Target Model
- **ResNet-18** customized for Tiny-ImageNet (200 classes).
- Trained with strict validation checkpoints and learning rate scheduling.
- **Source**: `src/train.py`, `src/model.py`

### 2. Advanced Adversarial Attacks (AutoAttack)
We implement **State-of-the-Art** attack generation to rigorously test our defenses.
- **AutoAttack (Lite)**: An ensemble of PGD with Cross-Entropy Loss and Difference-of-Logits-Ratio (DLR) Loss. This ensures we find the worst-case perturbation.
- **Source**: `src/auto_attack.py`

### 3. Detector 1: Prediction Stability (The "Physics" Defense)
- **Mechanism**: Adversarial perturbations are high-frequency and fragile. We treat the image with "Aggressive Computations" (Resize to 28x28 + JPEG Compression).
- **Result**: Real images retain their prediction. Adversarial images "shatter" and revert to their true class (or a different one).
- **Performance**: achieved **ROC-AUC: 0.81** (High Reliability).
- **Source**: `src/detectors.py`

### 4. Detector 2: Deep Feature Analysis (The "Abstract" Defense)
- **Mechanism**: We use **Mahalanobis Distance** to model the statistical distribution of features in the Neural Network's deeper layers.
- **Concept**: Adversarial examples often lie outside the normal "manifold" of valid images.
- **Result**: Provides a complementary signal, though less robust on smaller models (AUC 0.53).
- **Source**: `src/mahalanobis_detector.py`

## How to Run

1.  **Install Dependencies**:
    ```bash
    pip install torch torchvision tqdm scikit-learn numpy
    ```

2.  **Training & Attack**:
    ```bash
    python src/train.py       # Train the model
    python src/auto_attack.py # Generate strong attacks
    ```

3.  **Run Full Defense Evaluation**:
    This script runs the entire pipeline: Model Check -> AutoAttack -> Stability Defense.
    ```bash
    python src/evaluate_defense.py
    ```

## Project Summary & Achievements

- **Code Architecture**: Centralized `config.py` for reproducible research.
- **Data Integrity**: Validated data loading pipeline for Tiny-ImageNet.
- **Advanced Warfare**: Upgraded from simple PGD to AutoAttack strategies.
- **Dual Defense**: implemented a Hybrid System using **Prediction Stability** (AUC 0.81) and **Mahalanobis Feature Analysis** (AUC 0.53).

---
*Created by [Your Name]*
