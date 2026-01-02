# Implementation Plan - Phase 1 & 2: Cleanup and Restructuring

## Goal
To transform the current "spaghetti" codebase into a structured, modular, and reliable research framework. We will fix critical data handling errors (missing validation set) and eliminate hardcoded paths.

## User Review Required
> [!IMPORTANT]
> **Data Organization**: We are changing how data is loaded. We will ensure the Tiny-ImageNet validation folder is organized by class (using `fix_val_folder.py` logic) so standard PyTorch loaders work.

> [!WARNING]
> **Breaking Changes**: `train.py`, `dataset.py`, and `model.py` will be significantly refactored. Old scripts may not work without updates.

## Proposed Changes

### 1. Configuration Management
Create a central configuration file to remove hardcoded paths and magic numbers.

#### [NEW] [config.py](file:///Users/nellkhoury/Desktop/Adversarial_Project/src/config.py)
- Will contain:
    - `DATA_DIR`, `MODEL_DIR`, `OUTPUT_DIR`
    - `BATCH_SIZE`, `NUM_WORKERS`, `DEVICE`
    - `Hyperparameters` (LR, Epochs)
    - `Attack Parameters` (Epsilon, Alpha, Steps)

### 2. Data Handling
Fix the critical flaw of training on the dataset without validation.

#### [MODIFY] [dataset.py](file:///Users/nellkhoury/Desktop/Adversarial_Project/src/dataset.py)
- Import constants from `config.py`.
- Add `get_train_val_dataloaders()` function.
- Integrate `fix_val_folder` logic to ensure the validation set is usable.
- Return `train_loader` and `val_loader`.

### 3. Training Loop
Implement a proper training loop that evaluates on validation data.

#### [MODIFY] [train.py](file:///Users/nellkhoury/Desktop/Adversarial_Project/src/train.py)
- Use `config.py` for parameters.
- accepting arguments for resuming training.
- **Critical Fix**: Save model only when *Validation Loss* improves, not Training Loss.
- Add better logging (tqdm + final summary).

### 4. Adversarial Utilities
Clean up the attack and detection modules.

#### [MODIFY] [attack.py](file:///Users/nellkhoury/Desktop/Adversarial_Project/src/attack.py)
- Make `pgd_attack` a pure function dependent on config.
- Separate the "generate samples" script into a `main` block that calls the function.

## Verification Plan

### Automated Tests
We will run the following commands to verify the refactoring:

1.  **Check Data Loading**:
    ```bash
    python src/dataset.py
    # Should print: "Train batches: X, Val batches: Y"
    ```

2.  **Dry Run Training**:
    ```bash
    python src/train.py --dry-run
    # Should run 1 epoch (or few steps) and save a dummy model without crashing.
    ```

### Manual Verification
- Inspect `task.md` to ensure items are checked off.
- Review `config.py` to ensure all paths are correct for your machine.
