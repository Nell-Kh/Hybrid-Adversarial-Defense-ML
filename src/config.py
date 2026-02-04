import os
import torch

# PROJECT PATHS 
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
# PRO TIP: This logic makes the project portable across Mac (MPS), Linux/Windows (CUDA), and standard CPU.
if torch.cuda.is_available():
    DEVICE = torch.device("cuda") # NVIDIA GPUs (Linux/Windows)
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # Apple Silicon (Mac M1/M2/M3)
else:
    DEVICE = torch.device("cpu")  # Standard Fallback
NUM_WORKERS = 4  # Adjust based on your Mac's core count

# DATASET SETTINGS 
DATASET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
BATCH_SIZE = 64
IMG_SIZE = 64
NUM_CLASSES = 200

# TRAINING HYPERPARAMETERS 
LEARNING_RATE = 0.001
NUM_EPOCHS = 15
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "resnet_tinyimagenet.pth")

# ATTACK PARAMETERS (PGD) 
# The Core Constraint: We only allow 8/255 (approx 3%) pixel change. This ensures the attack remains invisible to humans.
ATTACK_EPSILON = 8/255
ATTACK_ALPHA = 2/255
ATTACK_STEPS = 10
