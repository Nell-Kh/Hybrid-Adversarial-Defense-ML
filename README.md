# Hybrid Adversarial Defense System

A robust framework for detecting and defending against adversarial attacks on image classifiers using Tiny-ImageNet.

## Project Structure

- **`src/`**: Core source code (training, attacks, detectors).
- **`docs/`**: Project documentation, plans, and walkthroughs.
- **`data/`**: Dataset storage (ignored by git).
- **`models/`**: Saved model checkpoints (ignored by git).
- **`outputs/`**: Generated adversarial samples and logs (ignored by git).

## Getting Started

1.  **Install Dependencies**:
    ```bash
    pip install torch torchvision tqdm matplotlib scikit-learn
    ```

2.  **Train the Model**:
    ```bash
    python src/train.py
    ```

3.  **Generate Attacks**:
    ```bash
    python src/attack.py
    ```

For full details, see the [Syllabus](docs/task.md) and [Walkthrough](docs/walkthrough.md).
