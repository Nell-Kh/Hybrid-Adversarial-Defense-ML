import torch
import torch.nn.functional as F
import os
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from model import get_model

# --- PRO CONFIGURATION ---
# We test 3 different types of physical stress
TESTS = {
    'noise': {'std': 0.02, 'iters': 10},
    'blur':  {'kernel': 3, 'iters': 1},     # Optical Blur (3x3 kernel)
    'rotate': {'degrees': 5, 'iters': 1}    # Geometric Rotation (+/- 5 degrees)
}

def get_image(path, device):
    t = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    try:
        img = Image.open(path).convert('RGB')
        return t(img).unsqueeze(0).to(device)
    except:
        return None

def check_noise_stability(model, image, base_pred, std, iters):
    """Test 1: Does it survive random pixel static?"""
    consistent = 0
    for _ in range(iters):
        noise = torch.randn_like(image) * std
        pred = model(image + noise).argmax(1).item()
        if pred == base_pred: consistent += 1
    return consistent / iters

def check_blur_stability(model, image, base_pred, kernel_size):
    """Test 2: Does it survive optical blurring? (Destroys high-freq attacks)"""
    # We apply a slight blur. 
    # Real objects (birds) stay birds when blurry. 
    # Fake objects (noise patterns) usually disappear.
    blurred_img = TF.gaussian_blur(image, kernel_size)
    pred = model(blurred_img).argmax(1).item()
    return 1.0 if pred == base_pred else 0.0

def check_rotation_stability(model, image, base_pred, degrees):
    """Test 3: Does it survive geometric rotation?"""
    # Rotate slightly clockwise
    rot_img = TF.rotate(image, degrees)
    pred1 = model(rot_img).argmax(1).item()
    
    # Rotate slightly counter-clockwise
    rot_img_2 = TF.rotate(image, -degrees)
    pred2 = model(rot_img_2).argmax(1).item()
    
    # Return average stability (0.0, 0.5, or 1.0)
    score = (int(pred1 == base_pred) + int(pred2 == base_pred)) / 2
    return score

def run_advanced_detector():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🔬 Running Multi-Modal Stress Test on {device}...")

    model = get_model(device)
    model.load_state_dict(torch.load("../models/resnet_tinyimagenet.pth", map_location=device))
    model.eval()

    clean_dir = "../data/clean_samples"
    adv_dir = "../data/adversarial_samples"
    
    print(f"{'ID':<4} | {'Type':<6} | {'Noise':<6} | {'Blur':<6} | {'Rot':<6} | {'Composite Score'}")
    print("-" * 65)

    for i in range(10): # Test first 10 images
        c_path = os.path.join(clean_dir, f"img_{i}.png")
        a_path = os.path.join(adv_dir, f"img_{i}.png")
        
        if not os.path.exists(c_path): break

        # We test both the CLEAN and the ATTACK version for each ID
        for img_type, path in [("Real", c_path), ("Fake", a_path)]:
            img = get_image(path, device)
            
            # 1. Get Baseline Prediction
            base_pred = model(img).argmax(1).item()

            # 2. Run The Stress Tests
            s_noise = check_noise_stability(model, img, base_pred, TESTS['noise']['std'], TESTS['noise']['iters'])
            s_blur  = check_blur_stability(model, img, base_pred, TESTS['blur']['kernel'])
            s_rot   = check_rotation_stability(model, img, base_pred, TESTS['rotate']['degrees'])

            # 3. Calculate Composite Score (Average of all physics tests)
            final_score = (s_noise + s_blur + s_rot) / 3

            # Print simple table row
            print(f"{i:<4} | {img_type:<6} | {s_noise:<6.2f} | {s_blur:<6.2f} | {s_rot:<6.2f} | {final_score:.2f}")

        print("-" * 65)

if __name__ == "__main__":
    run_advanced_detector()