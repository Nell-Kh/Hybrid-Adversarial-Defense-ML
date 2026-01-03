# Project Syllabus: Hybrid Adversarial Defense

You are currently failing "Code Organization 101". We are going to fix that. Here is your syllabus.

- [x] **Phase 1: Code Audit & Cleanup** <!-- id: 0 -->
    - [x] Review `model.py` and `dataset.py` for foundational issues. <!-- id: 1 -->
    - [x] Audit `train.py` and `evaluate.py` for training loop clarity. <!-- id: 2 -->
    - [x] Check `attack.py` for proper PGD implementation. <!-- id: 3 -->
    - [x] Assess `detectors.py` and `activation_detector.py` for redundancy and logic. <!-- id: 4 -->
- [x] **Phase 2: Restructuring** <!-- id: 5 -->
    - [x] Standardize configuration (stop hardcoding paths!). <!-- id: 6 -->
    - [x] Modularize training loops (Trainer class). <!-- id: 7 -->
    - [x] Ensure specific directories exist (`data`, `checkpoints`, `outputs`). <!-- id: 8 -->
- [x] **Phase 3: Implementation Verification** <!-- id: 9 -->
    - [x] Verify Prediction Stability Detector. <!-- id: 10 -->
    - [ ] Verify Activation-Based Detector. <!-- id: 11 -->
    - [ ] Verify Image Correction (JPEG + TVM). <!-- id: 12 -->
- [ ] **Phase 4: Final Exam (Visualization)** <!-- id: 13 -->
    - [ ] Ensure `visualize.py` produces correct Grad-CAM outputs. <!-- id: 14 -->
