
# Adversarial Attacks & Defense

This project explores the vulnerability of deep learning models (ResNet-18) to adversarial attacks and tests different defense mechanisms. 
We used the **Tiny-ImageNet** dataset (200 classes) to simulate a real-world classification task.

The goal is to understand how easily AI models can be tricked and how we can make them more robust.

---

## Project Structure

The code is organized into three main parts:
1.  **Attacks** (`src/attacks/`): Scripts that generate adversarial examples (noisy images).
2.  **Defense** (`src/defenses/`): Methods to detect or resist these attacks.
3.  **Visualization** (`src/visualize_all.py`): Tools to display the results.

### 1. Attacks Implemented
We tested four different attack methods, ranging from simple to advanced:
*   **PGD**: A standard "brute force" attack that changes pixel values to maximize error.
*   **DeepFool**: Finds the smallest possible change needed to flip the label.
*   **Carlini-Wagner (CW)**: A high-confidence optimization attack (L2).
*   **Adaptive Attack**: A custom attack we wrote that tries to fool both the model and our detection system simultaneously.

### 2. Defenses Implemented
We implemented two defenses proposed in recent research:
*   **Adversarial Training**: We re-trained the model on attacked images so it learns to recognize them.
*   **Mahalanobis Detector**: A statistical method that looks at the internal layers of the network to detect "abnormal" activity.

---

## How to Run

### Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running the Evaluation
To run all attacks and calculate the success rate:
```bash
python3 src/evaluate_suite.py --attacks all
```

### Visualizing Results
To save a comparison image of Clean vs. Attacked images:
```bash
python3 src/visualize_all.py
```
*Output will be saved in `data/viz_results/`.*

---

## 📊 Results Summary
*   **Standard Model**: ~60% accuracy on clean images, but 0% accuracy against strong attacks.
*   **Robst Model**: Retains accuracy even when attacked.
*   **Detector**: Successfully flags ~80% of adversarial examples.

---
*Nell Khoury, Celine michael *
