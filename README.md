# Hybrid Adversarial Defense System for Tiny-ImageNet

Silin Michael, Nell Khoury

*Date:* November 2025

## Project Overview
This project focuses on designing a hybrid defense system that detects and mitigates adversarial attacks on image classification models. The objective is to develop a method that can reliably identify adversarially perturbed images and apply corrective transformations to improve the model's robustness.

The system uses a ResNet-18 classifier trained on the Tiny-ImageNet dataset and incorporates two complementary detection strategies, followed by a lightweight image-restoration step.

## Methodology

### 1. Target Model and Adversarial Data Generation
The foundation of the system is a ResNet-18 model trained using transfer learning. The model is fine-tuned on the Tiny-ImageNet dataset, which consists of 200 classes of 64x64 images.

After training the classifier, adversarial examples are generated using iterative gradient-based attacks, specifically Projected Gradient Descent (PGD). These adversarial samples serve as training data for the detection module, enabling the system to distinguish between clean and perturbed inputs.

### 2. Hybrid Detection Module
The defense mechanism employs two independent detectors to ensure reliability.

* *Detector 1: Prediction Stability Analysis*
    This detector evaluates the sensitivity of the classifier to small, benign perturbations. Clean images typically produce consistent predictions under minimal input changes, whereas adversarial examples often exhibit unstable outputs.

* *Detector 2: Activation-Based DNN Classifier*
    This detector operates on the internal activations of the target model. We extract feature vectors from deep layers of the ResNet-18 and train a secondary neural network (DNN) to classify these vectors as originating from either clean or adversarial inputs.

### 3. Image Correction Module
Inputs flagged as adversarial are passed through a correction stage. Two preprocessing techniques are evaluated:
* *JPEG Compression:* Applied to suppress high-frequency perturbations introduced by attacks.
* *Total Variation Minimization (TVM):* Reduces high-frequency noise while preserving structural details.

## Visualization
To help interpret the model's behavior, the system generates Grad-CAM heatmaps that highlight the regions influencing the classifier's decisions. These visualizations provide qualitative insight into how adversarial perturbations affect the model and how the correction stage changes the focus of the network.
