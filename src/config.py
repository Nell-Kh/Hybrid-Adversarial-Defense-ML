import os
import torch

# --- PROJECT PATHS ---
# Base directory is the parent of 'src' (i.e., the project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CLEAN_SAMPLES_DIR = os.path.join(DATA_DIR, "clean_samples")
ADV_SAMPLES_DIR = os.path.join(DATA_DIR, "adversarial_samples")

# Ensure directories exist
for d in [DATA_DIR, MODEL_DIR, OUTPUT_DIR, CLEAN_SAMPLES_DIR, ADV_SAMPLES_DIR]:
    os.makedirs(d, exist_ok=True)

# --- SYSTEM SETTINGS ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
NUM_WORKERS = 4  # Adjust based on your Mac's core count

# --- DATASET SETTINGS ---
DATASET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
BATCH_SIZE = 64
IMG_SIZE = 64
NUM_CLASSES = 200

# --- TRAINING HYPERPARAMETERS ---
LEARNING_RATE = 0.001
NUM_EPOCHS = 15
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "resnet_tinyimagenet.pth")

# --- ATTACK PARAMETERS (PGD) ---
ATTACK_EPSILON = 8/255
ATTACK_ALPHA = 2/255
ATTACK_STEPS = 10
