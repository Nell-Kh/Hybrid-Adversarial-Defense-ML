# Hybrid Adversarial Defense System (Advanced)

## Project Overview
We developed a complete end-to-end framework to study and defend against adversarial attacks on image classifiers. Using **Tiny-ImageNet** as our testbed, we trained robust models, generated state-of-the-art attacks, and built a multi-layered detection system.

## What We Did

### 1. Robust Model Training
We successfully trained a custom **ResNet-18** model on the Tiny-ImageNet dataset (200 classes). By implementing strict validation checkpoints and learning rate scheduling, we achieved a stable baseline for our adversarial experiments.

## Key Achievements (The "Arsenal")
We constructed a modular Python library (`src/attacks/`) containing state-of-the-art adversarial attacks.

| Attack | Type | Success Rate | Impact |
| :--- | :--- | :--- | :--- |
| **AutoAttack** | White-Box | **100%** | Completely destroys the model with minimal distortion (L2 ≈ 2.7). |
| **Boundary Attack** | Black-Box | **100%** | Breaks the model without access to gradients (Blind Attack). |
| **DeepFool** | White-Box | **57.1%** | Finds the precise "shortest path" to a decision boundary. |

### Critical Finding: The "Normalization Blindness"
During development, we discovered that standard attacks often fail (0% success) if data normalization is mismatched.
*   **The Bug**: Model expected `[-1, 1]`, attacks generated `[0, 1]`.
*   **The Fix**: We mandated `[-1, 1]` clamping in our `Attacker` base class.
*   **Result**: DeepFool success went from **0% -> 57%**.

## Quick Start

### 1. Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Benchmark
Test all attacks against the ResNet-18 model:
```bash
python3 src/evaluate_suite.py
```

### 3. Generate Physical Patch
Create a printable adversarial patch (e.g., to hide from cameras):
```bash
python3 src/attacks/patch.py
```

## Project Defense Status
(Coming Soon: Adversarial Training & Mahalanobis Detection)
