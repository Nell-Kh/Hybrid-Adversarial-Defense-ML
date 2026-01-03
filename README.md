# Hybrid Adversarial Defense System

This project focuses on designing a hybrid defense system that detects and mitigates adversarial attacks on image classification models. The goal is to develop a method that can reliably identify adversarial perturbations and apply corrective transformations to improve the model's robustness.

Our work uses a **ResNet-18 classifier** trained on the **Tiny-ImageNet dataset** and incorporates two complementary detection strategies, followed by a lightweight image-restoration step.

## 1. Target Model and Adversarial Data Generation
We begin by training a ResNet-18 model using **transfer learning**. The model is fine-tuned on the Tiny-ImageNet dataset (200 classes).
- **Source Code**:
    - `src/model.py`: Defines the ResNet-18 architecture.
    - `src/train.py`: Training loop with validation and checkpointing.
    - `src/dataset.py`: Data loading and standardization.
    - `src/config.py`: Central configuration for all parameters.

Adversarial examples are generated using **Projected Gradient Descent (PGD)**.
- **Source Code**:
    - `src/attack.py`: Implementation of PGD attack generation.

## 2. Hybrid Detection Module
The defense system includes two independent detectors to improve reliability.

### Detector 1: Prediction Stability Analysis
Evaluates the sensitivity of the classifier to small, benign perturbations. Real images are stable; adversarial images are unstable.
- **Source Code**: `src/detectors.py` (Implements `get_anomaly_score` based on KL-Divergence).

### Detector 2: Activation-Based DNN Classifier
Operates on **internal activations** of the network. We extract features from a deep layer and train a small classifier to distinguish clean vs. adversarial features.
- **Source Code**: `src/activation_detector.py` (Feature extraction and Logistic Regression detector).

## 3. Image Correction Module
When an input is flagged as adversarial, it is passed through a correction stage:
- **JPEG Compression**: Suppresses high-frequency perturbations.
- **Total Variation Minimization (TVM)**: Reduces noise while preserving structure.
- **Source Code**: `src/detectors.py` (Contains transform logic for JPEG/Resizing).

## Project Documentation (`docs/`)
- **[Syllabus (task.md)](docs/task.md)**: The step-by-step plan we followed to clean and refactor the code.
- **[Walkthrough](docs/walkthrough.md)**: Proof of verification and dry-run results.
- **[Implementation Plan](docs/implementation_plan.md)**: Technical details of the refactoring process.

## Getting Started
1.  **Install Dependencies**: `pip install torch torchvision tqdm scikit-learn`
2.  **Train Target Model**: `python src/train.py`
3.  **Generate Attacks**: `python src/attack.py`
