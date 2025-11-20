# Hybrid Adversarial Defense System for Tiny-ImageNet

**Authors:** Silin Michael, Nell Khoury (November 20, 2025)

## Project Goal: Building a Super Smart Guard

Our project is all about creating a system that can **catch and fix sneaky adversarial attacks** that try to fool AI models. Instead of just letting the model guess wrong, our system will act like a security guard: it detects the attack and then cleans the image so the model can make the right prediction.

* **The Model We Protect:** We'll use a **ResNet-18** model.
* **The Data We Use:** We're working with the more complex **Tiny-ImageNet** dataset (200 types of objects, $64\times64$ images).
* **The Big Idea:** We are combining two different detectors to make the defense much stronger.

## How the System Works (The Three Main Parts)

### 1. Setting Up the Target
First, we train our main model, the ResNet-18. We use **Transfer Learning** to save time and make it a good classifier on the Tiny-ImageNet data. Once it's trained, we create tons of fake attack images (**Adversarial Examples**) using methods like PGD. These fake images will be used to teach our security system what an attack looks like.

### 2. The Hybrid Detector
This is the core of our project, combining two ways to spot trouble:

* **Detector 1: The Stability Check:** We check how stable the model's guess is. If we make a tiny, innocent change to an image, the prediction shouldn't flip completely. If it does, the image is probably corrupt.
* **Detector 2: The Inner Mind Reader (Activations DNN):** We look inside the Target Model, taking the numerical output from a deep layer (**Activations**). We then train a separate, smaller **Neural Network (DNN)** to look *only* at these internal numbers and decide if they came from a clean image or an attacked one. This lets the system automatically learn the hidden patterns of an attack.

### 3. Active Correction (The Fixer)
If our Hybrid Detector says "Warning! Attack Detected!", the system takes action:

* **The Fix Options:** We apply a simple cleaning transformation.
    * **Option 1: JPEG Compression:** Applying slight JPEG Compression destroys the very weak, invisible adversarial noise.
    * **Option 2: Total Variation Minimization (TVM):** TVM smooths the image while keeping important edges sharp, which helps remove the high-frequency adversarial noise.
* **The Test:** We send the cleaned image back to the Target Model. If the classification is now correct, the defense worked!

## Visualization and Proof
Finally, to show exactly *where* the attack was, we'll generate a **Heatmap** using a tool like **Grad-CAM**. This map will highlight the exact pixels that the model was looking at when it made the wrong prediction, visually proving the source of the problem.
