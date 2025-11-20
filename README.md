# Hybrid Adversarial Defense System for Tiny-ImageNet

**Authors:** Silin Michael, Nell Khoury (November 20, 2025)

## Project Goal: Building a Super Smart Guard Dog

[cite_start]Our project is all about creating a system that can **catch and fix sneaky adversarial attacks** that try to fool AI models[cite: 37]. [cite_start]Instead of just letting the model guess wrong, our system will act like a security guard: it detects the attack and then cleans the image so the model can make the right prediction[cite: 38].

* [cite_start]**The Model We Protect:** We'll use a **ResNet-18** model[cite: 39].
* [cite_start]**The Data We Use:** We're working with the more complex **Tiny-ImageNet** dataset (200 types of objects, $64\times64$ images)[cite: 40].
* [cite_start]**The Big Idea:** We are combining two different detectors to make the defense much stronger[cite: 41].

## How the System Works (The Three Main Parts)

### 1. Setting Up the Target
First, we train our main model, the ResNet-18. [cite_start]We use **Transfer Learning** to save time and make it a good classifier on the Tiny-ImageNet data[cite: 44]. [cite_start]Once it's trained, we create tons of fake attack images (**Adversarial Examples**) using methods like PGD[cite: 45]. [cite_start]These fake images will be used to teach our security system what an attack looks like[cite: 46].

### 2. The Hybrid Detector
[cite_start]This is the core of our project, combining two ways to spot trouble[cite: 48]:

* [cite_start]**Detector 1: The Stability Check:** We check how stable the model's guess is[cite: 49]. [cite_start]If we make a tiny, innocent change to an image, the prediction shouldn't flip completely[cite: 50]. [cite_start]If it does, the image is probably corrupt[cite: 51].
* [cite_start]**Detector 2: The Inner Mind Reader (Activations DNN):** We look inside the Target Model, taking the numerical output from a deep layer (**Activations**)[cite: 53, 54]. [cite_start]We then train a separate, smaller **Neural Network (DNN)** to look *only* at these internal numbers and decide if they came from a clean image or an attacked one[cite: 55]. [cite_start]This lets the system automatically learn the hidden patterns of an attack[cite: 56].

### 3. Active Correction (The Fixer)
[cite_start]If our Hybrid Detector says "Warning! Attack Detected!", the system takes action[cite: 58]:

* [cite_start]**The Fix Options:** We apply a simple cleaning transformation[cite: 59].
    * [cite_start]**Option 1: JPEG Compression:** Applying slight JPEG Compression destroys the very weak, invisible adversarial noise[cite: 60, 61].
    * [cite_start]**Option 2: Total Variation Minimization (TVM):** TVM smooths the image while keeping important edges sharp, which helps remove the high-frequency adversarial noise[cite: 62].
* **The Test:** We send the cleaned image back to the Target Model. [cite_start]If the classification is now correct, the defense worked[cite: 63, 64]!

## Visualization and Proof
[cite_start]Finally, to show exactly *where* the attack was, we'll generate a **Heatmap** using a tool like **Grad-CAM**[cite: 66]. [cite_start]This map will highlight the exact pixels that the model was looking at when it made the wrong prediction, visually proving the source of the problem[cite: 67].
