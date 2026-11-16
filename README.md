# Visualizing Adversarial Robustness: Attacks & Defenses in Deep Learning

This repository contains the source code for our adversarial machine learning project. We evaluate how vulnerable a standard ResNet-18 is to adversarial perturbations, and test several defenses against them.

We use the **Tiny-ImageNet** dataset (200 classes).

---

## Interactive Streamlit Dashboard

The project includes an interactive Streamlit dashboard for generating and visualizing adversarial attacks.

### Dashboard Features:
1. **Live evaluation:** select any image from the Tiny-ImageNet validation set and apply 7 different attacks.
2. **Targeted attacks:** push the network toward a user-chosen target class (e.g., classifying a dog as a sports car).
3. **Perturbation metrics:** $L_2$ and $L_\infty$ norms, SSIM and PSNR, to measure how visible the perturbation is.
4. **Benchmark suite:** runs 4 main attacks on both the standard and the robust model and shows the results as radar charts, with a PDF report export.
5. **Loss landscape:** a 3D visualization of the loss around an input, to illustrate how attacks move it across the decision boundary.
6. **Patch attacks:** drag an adversarial patch anywhere on the image to simulate a physical-world attack.

---

## Main Results

1. **Standard CNNs are highly vulnerable.** The baseline ResNet-18 reaches $78.12\%$ clean accuracy, but its robust accuracy drops to $0.0\%$ under an $L_\infty$ PGD attack ($\epsilon=8/255$).
2. **TRADES trades clean accuracy for robustness.** TRADES lowers clean accuracy to $70.64\%$ but raises robust accuracy under the same PGD attack to $43.37\%$.
3. **Layered defenses add incremental robustness.** Against AutoAttack, TRADES alone keeps $32.6\%$ robust accuracy. In our ablation, the post-hoc components (stochastic ensemble: $8.1\%$, autoencoder: $12.4\%$) add incremental robustness on top of TRADES, and the full pipeline reaches $38.2\%$.
4. **Static defenses can be bypassed by adaptive attacks.** Our adaptive PGD attack, which also minimizes the detector's anomaly score, evades the Mahalanobis-distance detector on $34\%$ of samples.

---

## Implemented Attack Algorithms

The framework supports a comprehensive suite of attack vectors:

1. **FGSM (Fast Gradient Sign Method):** $L_\infty$ bounded, single-step gradient approximation. Good for fast, brute-force evaluation.
2. **DeepFool:** An iterative $L_2$ attack that computes the minimal possible perturbation required to push the sample across the closest linear approximation of the decision boundary.
3. **C&W (Carlini-Wagner $L_2$):** A highly potent, optimization-based attack that solves a bounded box constraint problem using a margin-based objective function. Supports Targeted mode.
4. **Adaptive PGD (Ninja):** A custom, defense-aware attack. Modifies the traditional PGD objective function to simultaneously maximize classification error while *minimizing* detector anomaly scores.
5. **AutoAttack:** A parameter-free ensemble of state-of-the-art attacks (APGD-ce, APGD-dlr, FAB, Square). Used as the gold-standard benchmark in academic literature.
6. **Boundary Attack:** A black-box, decision-based attack. Starts from an adversarial target and performs a constrained random walk along the decision boundary, requiring zero gradient information.
7. **Expectation over Transformation (EoT):** An advanced attack designed specifically to defeat stochastic/randomized defenses by computing the expected gradient across a distribution of noise transformations.

---

## Implemented Defense Mechanisms

1. **Robust Adversarial Training (TRADES):** The core defense. Re-trains the ResNet-18 using the TRADES algorithm (Zhang et al. 2019), which mathematically optimizes the trade-off between clean accuracy and boundary robustness using a KL-Divergence regularizer.
2. **Mahalanobis Distance Detector:** Analyzes the internal deep representations (activations) of the neural network to identify out-of-distribution adversarial vectors before they reach the final softmax layer.
3. **Stochastic Ensemble (TTA):** A randomized inference defense. It applies random spatial jitter to the incoming image and evaluates 10 micro-variations, utilizing a majority-vote consensus to destroy overly brittle adversarial grid patterns.
4. **Neural Autoencoder Cleaner:** An auxiliary UNet-style architecture trained specifically to denoise adversarial inputs. It attempts to project the adversarial sample back onto the clean data manifold prior to classification.

---

## Installation & Execution

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

*Note: All pre-trained model weights (including the standard ResNet-18, the TRADES-fortified model, and the Neural Autoencoder) are explicitly included in the `models/` directory of this repository. You do not need to train the models from scratch to run the interactive dashboard.*

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
Nell Khoury, Celine Michael*
