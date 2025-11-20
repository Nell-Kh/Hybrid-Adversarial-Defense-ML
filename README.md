# Hybrid Adversarial Defense System for Tiny-ImageNet

**Authors:** Silin Michael, Nell Khoury (November 20, 2025)

## Project Overview

This project focuses on designing a hybrid defense system that detects and mitigates adversarial attacks on image classification models. The goal is to develop a method that can reliably identify adversarially perturbed images and apply corrective transformations that improve the model’s robustness.

Our work uses a **ResNet-18 classifier** trained on the **Tiny-ImageNet dataset** and incorporates two complementary detection strategies, followed by a lightweight image-restoration step.

## 1. Target Model and Adversarial Data Generation

We begin by training a ResNet-18 model using **transfer learning**. The model is fine-tuned on the Tiny-ImageNet dataset, which contains 200 classes of 64x64 images.

After training the classifier, we generate adversarial examples using iterative gradient-based attacks such as **Projected Gradient Descent (PGD)**. These adversarial samples serve as training data for the detection module and allow the system to learn the distinction between clean and perturbed inputs.

## 2. Hybrid Detection Module

The defense system includes two independent detectors whose outputs are combined to improve reliability.

### Detector 1: Prediction Stability Analysis

This detector evaluates the sensitivity of the classifier to small, benign perturbations. Clean images typically produce consistent predictions under minimal input changes, while adversarial examples often lead to unstable or inconsistent outputs. The degree of prediction variation is used as a signal for identifying suspicious inputs.

### Detector 2: Activation-Based DNN Classifier

The second detector operates on **internal activations** of the target model rather than on the image itself. We extract features from a deep layer of the ResNet-18 and train a small neural network (**DNN**) to classify these activation vectors as originating from either clean or adversarial inputs. This approach leverages the internal representation learned by the model and enables the detector to capture subtle patterns characteristic of adversarial perturbations.

## 3. Image Correction Module

When an input is flagged as potentially adversarial, it is passed through a correction stage. We evaluate two preprocessing-based techniques:

* **JPEG Compression:** Mild compression is applied to suppress high-frequency perturbations commonly introduced by adversarial attacks.
* **Total Variation Minimization (TVM):** This method reduces high-frequency noise while preserving important structural details in the image.

After correction, the restored image is re-evaluated by the target classifier to determine whether the prediction is recovered.

## Visualization

To help interpret the model’s behavior, we generate **Grad-CAM heatmaps** that highlight the regions influencing the classifier’s decisions. These visualizations provide qualitative insight into how adversarial perturbations affect the model and how the correction stage changes the focus of the network.
