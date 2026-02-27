# Visualizing Adversarial Robustness: Attacks & Defenses in Deep Learning

This repository contains the complete source code for our Adversarial Machine Learning research project. 
The objective of this project is to evaluate the vulnerability of standard Convolutional Neural Networks (ResNet-18) to adversarial perturbations, and to implement state-of-the-art mathematical defense mechanisms to fortify them.

We utilize the **Tiny-ImageNet** dataset (200 classes) to simulate a high-dimensional, real-world classification environment.

---

## 🚀 Interactive Streamlit Dashboard

The core of this project is a fully interactive, academic-grade Streamlit Web Dashboard that allows users to generate and visualize adversarial attacks in real-time.

### Dashboard Features:
1. **Live Evaluation Pipeline:** Select any image from the Tiny-ImageNet validation set and apply 7 different attack methodologies on the fly.
2. **Targeted Feature Spoofing:** Override untargeted attacks to mathematically force the network into hallucinating user-defined target classes (e.g., forcing the model to classify a dog as a sports car).
3. **Mathematical Stealth Metrics:** Live computation of $L_2$ Norm, $L_\infty$ Norm, Structural Similarity Index (SSIM), and Peak Signal-to-Noise Ratio (PSNR) to guarantee mathematical imperceptibility.
4. **Radar Benchmark Suite:** An automated gauntlet that runs 4 primary attacks across both standard and robust models simultaneously, generating interactive Plotly Radar Charts that prove geometric stability. Features a built-in PDF Report exporter.
5. **Interactive Loss Landscape:** Render a 3D topological map of the neural network's decision boundary to visualize how adversarial attacks exploit highly non-convex local maxima.
6. **Physical Patch Attacks:** Drag and drop an adversarial patch anywhere on the image to simulate physical-world camouflage attacks.

---

## ⚔️ Implemented Attack Algorithms

The framework supports a comprehensive suite of attack vectors:

1. **FGSM (Fast Gradient Sign Method):** $L_\infty$ bounded, single-step gradient approximation. Good for fast, brute-force evaluation.
2. **DeepFool:** An iterative $L_2$ attack that computes the minimal possible perturbation required to push the sample across the closest linear approximation of the decision boundary.
3. **C&W (Carlini-Wagner $L_2$):** A highly potent, optimization-based attack that solves a bounded box constraint problem using a margin-based objective function. Supports Targeted mode.
4. **Adaptive PGD (Ninja):** A custom, defense-aware attack. Modifies the traditional PGD objective function to simultaneously maximize classification error while *minimizing* detector anomaly scores.
5. **AutoAttack:** A parameter-free ensemble of state-of-the-art attacks (APGD-ce, APGD-dlr, FAB, Square). Used as the gold-standard benchmark in academic literature.
6. **Boundary Attack:** A black-box, decision-based attack. Starts from an adversarial target and performs a constrained random walk along the decision boundary, requiring zero gradient information.
7. **Expectation over Transformation (EoT):** An advanced attack designed specifically to defeat stochastic/randomized defenses by computing the expected gradient across a distribution of noise transformations.

---

## 🛡️ Implemented Defense Mechanisms

1. **Robust Adversarial Training (TRADES):** The core defense. Re-trains the ResNet-18 using the TRADES algorithm (Zhang et al. 2019), which mathematically optimizes the trade-off between clean accuracy and boundary robustness using a KL-Divergence regularizer.
2. **Mahalanobis Distance Detector:** Analyzes the internal deep representations (activations) of the neural network to identify out-of-distribution adversarial vectors before they reach the final softmax layer.
3. **Stochastic Ensemble (TTA):** A randomized inference defense. It applies random spatial jitter to the incoming image and evaluates 10 micro-variations, utilizing a majority-vote consensus to destroy overly brittle adversarial grid patterns.
4. **Neural Autoencoder Cleaner:** An auxiliary UNet-style architecture trained specifically to denoise adversarial inputs. It attempts to project the adversarial sample back onto the clean data manifold prior to classification.

---

## 🛠️ Installation & Execution

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launching the Interactive Dashboard
The easiest way to interact with the project is via the Streamlit UI:
```bash
python3 -m streamlit run src/app.py
```
*The dashboard will automatically open in your default browser at `http://localhost:8501`.*

### 3. Command Line Interfaces (CLI)
You can also run specific evaluations programmatically:

Evaluate the full suite (Terminal Output):
```bash
python3 src/evaluate_suite.py --attacks all
```

Train a new robust model (Requires GPU):
```bash
python3 src/train_advanced.py --arch resnet18 --epochs 40 --beta 6.0
```

---

*Authors: Nell Khoury, Celine Michael*
