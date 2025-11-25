# Hybrid Adversarial Defense System for Tiny-ImageNet

**Authors:** Silin Michael, Nell Khoury
**Date:** November 2025

## Project Overview
[cite_start]This project focuses on designing a hybrid defense system that detects and mitigates adversarial attacks on image classification models[cite: 1, 5]. [cite_start]The objective is to develop a method that can reliably identify adversarially perturbed images and apply corrective transformations to improve the model's robustness[cite: 6].

[cite_start]The system uses a ResNet-18 classifier trained on the Tiny-ImageNet dataset and incorporates two complementary detection strategies, followed by a lightweight image-restoration step[cite: 7].

## Methodology

### 1. Target Model and Adversarial Data Generation
[cite_start]The foundation of the system is a ResNet-18 model trained using transfer learning[cite: 9]. [cite_start]The model is fine-tuned on the Tiny-ImageNet dataset, which consists of 200 classes of 64x64 images[cite: 10].

[cite_start]After training the classifier, adversarial examples are generated using iterative gradient-based attacks, specifically Projected Gradient Descent (PGD)[cite: 11]. [cite_start]These adversarial samples serve as training data for the detection module, enabling the system to distinguish between clean and perturbed inputs[cite: 12].

### 2. Hybrid Detection Module
[cite_start]The defense mechanism employs two independent detectors to ensure reliability[cite: 14].

* **Detector 1: Prediction Stability Analysis**
    This detector evaluates the sensitivity of the classifier to small, benign perturbations. Clean images typically produce consistent predictions under minimal input changes, whereas adversarial examples often exhibit unstable outputs[cite: 15, 16].

* **Detector 2: Activation-Based DNN Classifier**
    This detector operates on the internal activations of the target model. We extract feature vectors from deep layers of the ResNet-18 and train a secondary neural network (DNN) to classify these vectors as originating from either clean or adversarial inputs[cite: 18, 19, 20].

### 3. Image Correction Module
Inputs flagged as adversarial are passed through a correction stage. Two preprocessing techniques are evaluated:
* [cite_start]**JPEG Compression:** Applied to suppress high-frequency perturbations introduced by attacks[cite: 24].
* [cite_start]**Total Variation Minimization (TVM):** Reduces high-frequency noise while preserving structural details[cite: 25].

