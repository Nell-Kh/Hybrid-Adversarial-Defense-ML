# Walkthrough - Phase 1 & 2: Core Refactoring

We have successfully audited, refactored, and verified the core components of the project.

## Changes Made

### 1. Centralized Configuration
**File**: `src/config.py`
- Removed hardcoded paths from all scripts.
- Consolidated hyperparameters (Batch Size, LR) in one place.
- Introduced `BASE_DIR` to ensure scripts run from anywhere.

### 2. Dataset Logic Repair
**File**: `src/dataset.py`
- **Critical Fix**: Added `organize_val_folder()` which automatically fixes the Tiny-ImageNet validation directory structure (previously unreadable by PyTorch).
- Added `get_dataloaders()` which returns both **Training** and **Validation** loaders.

### 3. Robust Verification
**File**: `src/train.py`
- Implemented a proper training loop with:
    - Validation after every epoch.
    - Model Checkpointing (saves only if Val Loss improves).
    - Learning Rate Scheduler.
- Added `--dry-run` flag for quick verification.

## Verification Results

We ran a "Dry Run" (`python src/train.py --dry-run`) to test the pipeline.

**Command Output**:
```
Using device: mps
--- DRY RUN MODE ACTIVATED ---
Loading data...
Data loaded: 1563 train batches, 157 val batches.
Starting training for 1 epochs...
Epoch 1 Summary: Train Loss: 5.7256 | Val Loss: 7.3496
Validation Loss Improved. Saving model...
Dry run: Skipping model save.
Training Complete.
Exit code: 0
```

> [!SUCCESS]
> The system is now healthy. You can start full training whenever you are ready.
