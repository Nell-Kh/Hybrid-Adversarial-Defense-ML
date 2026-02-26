import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Allow imports from src
sys.path.append(os.path.dirname(__file__))

# Allow imports from src regardless of where script is run
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
# Also add parent dir if needed for some configs
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import config
from model import get_model
from dataset import get_dataloaders
from attacks.auto_attack import AutoAttackLite
from attacks.deepfool import DeepFool
from attacks.boundary import BoundaryAttack
from attacks.cw import CWAttacker
from attacks.patch import PatchApplier
from attacks.adaptive import AdaptiveAttacker
from defenses.mahalanobis import MahalanobisDetector
from defenses.stochastic_ensemble import TTA_Ensemble
from defenses.image_cleaning import apply_cleaning
from utils_vis import GradCAM, apply_heatmap
from visualize_landscape import visualize_loss_landscape
from visualize_patch import get_saliency_map

# --- CONFIG ---
st.set_page_config(page_title="Adversarial Robustness Evaluation", layout="wide")
DEVICE = config.DEVICE

# --- CACHED LOADERS ---
@st.cache_resource
def load_models():
    # 1. Standard Model (Baseline)
    victim = get_model(DEVICE)
    victim_path = os.path.join("models", "resnet_tinyimagenet.pth")
    if os.path.exists(victim_path):
        victim.load_state_dict(torch.load(victim_path, map_location=DEVICE))
    else:
        st.error(f"Standard model not found at {victim_path}")
    victim.eval()
    
    # 2. Robust Model (Defense)
    hero = get_model(DEVICE)
    hero_path = os.path.join("models", "resnet_robust.pth")
    if os.path.exists(hero_path):
        checkpoint = torch.load(hero_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            hero.load_state_dict(checkpoint['model_state_dict'])
        else:
            hero.load_state_dict(checkpoint)
    else:
        st.warning(f"Robust model not found at {hero_path}. Falling back to standard model for robust predictions.")
        # Fallback so the app doesn't crash completely
        hero = victim
    hero.eval()
    
    # 3. Mahalanobis Detector (Iron Dome)
    detector = MahalanobisDetector(victim, DEVICE)
    detector.load_stats() # Attempts to load from disk
    
    # 4. Stochastic Ensemble (TTA / Randomized Smoothing)
    # Wraps the robust model to provide Test-Time Augmentation defenses
    tta_hero = TTA_Ensemble(hero, num_copies=10, max_shift=2, noise_std=0.02)
    
    return victim, hero, tta_hero, detector

@st.cache_resource
def get_val_dataset():
    # Fetch the full dataset object instead of just one batch
    # num_workers=0 is critical for Streamlit on macOS to avoid multiprocessing crash
    _, val_loader = get_dataloaders(batch_size=1, num_workers=0)
    return val_loader.dataset

@st.cache_data
def load_class_mapping():
    """Maps 0-199 PyTorch indices to human-readable TinyImageNet names."""
    wnid_to_name = {}
    words_file = os.path.join(config.DATA_DIR, "tiny-imagenet-200", "words.txt")
    if os.path.exists(words_file):
        with open(words_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    wnid_to_name[parts[0]] = parts[1]
    
    train_dir = os.path.join(config.DATA_DIR, 'tiny-imagenet-200', 'train')
    if os.path.exists(train_dir):
        wnids = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        # Often a class has multiple comma-separated names, we take the first
        idx_to_name = {i: wnid_to_name.get(wnid, wnid).split(',')[0] for i, wnid in enumerate(wnids)}
        return idx_to_name
    return {}

# --- UI ---
st.title("Adversarial Robustness Evaluation")
st.markdown("### Comparative Analysis: Standard ResNet18 vs. TRADES-Robust ResNet18")

# Create Tabs
tab_eval, tab_landscape, tab_patch = st.tabs(["Evaluation & Defense", "Loss Landscape", "Patch Attacks"])

victim, hero, tta_hero, detector = load_models()
val_dataset = get_val_dataset()
labels = val_dataset.targets # List of 10,000 integers
class_mapping = load_class_mapping()

# Helper to denormalize for display
def to_display(img_tensor):
    img = img_tensor.clone().detach() * 0.5 + 0.5
    img = img.clamp(0, 1)
    return img.squeeze().cpu().permute(1, 2, 0).numpy()

# Sidebar Control
with st.sidebar:
    st.header("Evaluation Controls")
    
    # 1. Group images by class for the selector
    class_to_indices = {}
    for idx, label_int in enumerate(labels):
        c_name = class_mapping.get(label_int, f"Class {label_int}")
        if c_name not in class_to_indices:
            class_to_indices[c_name] = []
        class_to_indices[c_name].append(idx)
        
    # Sort class names alphabetically for the dropdown
    sorted_class_names = sorted(list(class_to_indices.keys()))
    
    # Select Class
    selected_class = st.selectbox("1. Select Image Category", sorted_class_names, help="Pick a category (e.g., goldfish, bears, sports cars) to find images of that type.")
    
    # Get available indices for that class
    available_indices = class_to_indices[selected_class]
    
    # 2. Select Image Index
    # Show user-friendly 1-based index (Image 1, Image 2, etc.) mapped to the real index
    selected_local_idx = st.selectbox("2. Select Specific Image", range(len(available_indices)), format_func=lambda x: f"Image #{x+1}", help="Pick a specific image from the chosen category.")
    
    # Map back to the global tensor index
    img_idx = available_indices[selected_local_idx]
    
    # Removed the random slider since we now have a clean categorical dropdown
    
    st.divider()
    st.header("Attack Parameters")
    attack_name = st.selectbox("Adversarial Attack Method", [
        "None", 
        "FGSM (Fast Gradient)", 
        "DeepFool (Minimum Norm)", 
        "AutoAttack (Ensemble)", 
        "C&W (L2 Optimization)",
        "Ninja (Adaptive PGD)",
        "Boundary (Black-Box)"
    ])
    
    # Dynamic parameters
    epsilon = st.slider("Perturbation Magnitude (Epsilon)", 0.0, 0.1, 0.031, step=0.001, help="Controls how much noise the attacker is allowed to add. Higher = more visible noise but stronger attack.")
    
    cleaning_method = st.selectbox("Image Cleaning Defense", [
        "None", 
        "Gaussian Blur", 
        "Bit Depth Reduction (3-bit)",
        "Bit Depth Reduction (4-bit)",
        "Bit Depth Reduction (5-bit)",
        "Bit Depth Reduction (6-bit)",
        "Bit Depth Reduction (7-bit)",
        "Median Filter"
    ], help="Apply a preprocessing transformation to mathematically scrub adversarial noise from the image before inference.")
    
    if attack_name in ["AutoAttack (Ensemble)", "C&W (L2 Optimization)", "Ninja (Adaptive PGD)", "Boundary (Black-Box)"]:
        steps = st.slider("Optimization Steps", 10, 200, 50, step=10)
    
    st.divider()
    st.header("Visualization Options")
    show_heatmap = st.checkbox("Enable Grad-CAM Attention Maps")
    enable_tta = st.checkbox("Enable Stochastic Ensemble Defense", value=False, help="Generates 10 slightly noisy/shifted micro-variations of the incoming adversarial image. All 10 are fed to the robust model, destroying brittle adversarial grid patterns and exposing the true image via majority consensus.")
    
    if st.button("Run Analysis"):
        run_analysis = True
    else:
        run_analysis = False

# --- EVALUATION TAB ---
with tab_eval:
    # --- LOGIC ---
    # Fetch image and label on-the-fly from the dataset
    img_tensor, label_int = val_dataset[img_idx]
    target_image = img_tensor.unsqueeze(0).to(DEVICE)
    target_label = torch.tensor([label_int], dtype=torch.long).to(DEVICE)
    target_label_int = label_int
    
    # Get Classes
    class_name = class_mapping.get(target_label_int, f"Class {target_label_int}")
    
    # --- ATTACK EXECUTION ---
    adv_image = target_image.clone()
    noise = torch.zeros_like(target_image)
    
    if run_analysis and attack_name != "None":
        with st.spinner(f"Executing {attack_name} optimization..."):
            if attack_name == "FGSM (Fast Gradient)":
                target_image.requires_grad = True
                output = victim(target_image)
                loss = nn.CrossEntropyLoss()(output, target_label)
                victim.zero_grad()
                loss.backward()
                data_grad = target_image.grad.data
                sign_data_grad = data_grad.sign()
                adv_image = target_image + epsilon * sign_data_grad
                adv_image = torch.clamp(adv_image, -1, 1)
                target_image.requires_grad = False
                
            elif attack_name == "DeepFool (Minimum Norm)":
                attacker = DeepFool(victim, DEVICE, overshoot=0.02, max_iter=steps if 'steps' in locals() else 50)
                adv_image = attacker.attack(target_image, target_label)
                
            elif attack_name == "AutoAttack (Ensemble)":
                attacker = AutoAttackLite(victim, DEVICE, eps=epsilon)
                adv_image = attacker.attack(target_image, target_label)
                
            elif attack_name == "C&W (L2 Optimization)":
                attacker = CWAttacker(victim, DEVICE, steps=steps if 'steps' in locals() else 50)
                adv_image = attacker.attack(target_image, target_label)
                
            elif attack_name == "Ninja (Adaptive PGD)":
                attacker = AdaptiveAttacker(victim, DEVICE, detector, eps=epsilon, steps=steps if 'steps' in locals() else 50)
                adv_image = attacker.attack(target_image, target_label)
                
            elif attack_name == "Boundary (Black-Box)":
                attacker = BoundaryAttack(victim, DEVICE, steps=steps if 'steps' in locals() else 50)
                adv_image = attacker.attack(target_image, target_label)
                
        noise = (adv_image - target_image).abs()
        
    # --- IMAGE CLEANING ---
    if cleaning_method != "None":
        adv_image = apply_cleaning(adv_image, cleaning_method, DEVICE)
    
    # --- DISPLAY ---
    with torch.no_grad():
        pred_victim_logits = victim(adv_image)
        pred_victim = pred_victim_logits.argmax(1).item()
        
        # Branch defense evaluation based on UI Toggle
        if enable_tta:
            # We pass return_consensus=True to get the dictionary of votes
            pred_hero_logits, vote_breakdown = tta_hero(adv_image, return_consensus=True)
            pred_hero = pred_hero_logits.argmax(1).item()
        else:
            pred_hero_logits = hero(adv_image)
            pred_hero = pred_hero_logits.argmax(1).item()
            vote_breakdown = None
        
        # Calculate Trust Score
        try:
            trust_score_val = detector.get_trust_score(adv_image)[0] * 100  # Percentage
        except Exception as e:
            trust_score_val = 100.0 # Default to trusted if detector fails
    
    vis_victim = adv_image
    vis_hero = adv_image
    
    if show_heatmap:
        cam_v = GradCAM(victim, victim.layer4[-1])
        heatmap_v = cam_v(adv_image, pred_victim)
        overlay_v = apply_heatmap(adv_image * 0.5 + 0.5, heatmap_v)
        cam_h = GradCAM(hero, hero.layer4[-1])
        heatmap_h = cam_h(adv_image, pred_hero)
        overlay_h = apply_heatmap(adv_image * 0.5 + 0.5, heatmap_h)
    
    # Render Stats Summary
    st.subheader("Statistical Analysis (Mahalanobis Distance)")
    st.progress(int(trust_score_val))
    if trust_score_val < 50:
        st.warning(f"Anomalous input detected! Model Trust: {trust_score_val:.1f}%. This looks like an adversarial attack or out-of-distribution sample.")
    else:
        st.success(f"Input looks clean. Model Trust: {trust_score_val:.1f}%.")
        
    st.divider()

    # Pre-process noise for visualization (min-max normalization to make patterns visible)
    display_noise = noise.detach().squeeze().cpu().permute(1,2,0).numpy()
    noise_max = display_noise.max()
    if noise_max > 0:
        display_noise = display_noise / noise_max

    c1, c2, c3 = st.columns(3)
    with c1: st.image(to_display(target_image), caption=f"1. Original ({class_name})", use_container_width=True)
    with c2: st.image(display_noise, caption="2. Attack Noise (Amplified)", clamp=True, use_container_width=True)
    with c3: st.image(to_display(adv_image), caption="3. Adversarial Input", use_container_width=True)
    
    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Standard ResNet18 (Baseline)")
        pred_vic_name = class_mapping.get(pred_victim, f"Class {pred_victim}")
        if pred_victim == target_label_int: st.success(f"Prediction: CORRECT ({pred_vic_name})")
        else: st.error(f"Prediction: INCORRECT ({pred_vic_name})")
        if show_heatmap: st.image(overlay_v, caption="Baseline Attention Map", use_container_width=True)
            
    with col_b:
        st.subheader("TRADES ResNet18 (Robust)")
        pred_hero_name = class_mapping.get(pred_hero, f"Class {pred_hero}")
        if pred_hero == target_label_int: st.success(f"Prediction: CORRECT ({pred_hero_name})")
        else: st.warning(f"Prediction: INCORRECT ({pred_hero_name})")
        
        if enable_tta and vote_breakdown:
            st.markdown("**(Stochastic Ensemble Consensus Vote Tracker)**")
            for voted_class_idx, count in vote_breakdown.items():
                voted_name = class_mapping.get(voted_class_idx, f"Class {voted_class_idx}")
                pct = count / tta_hero.num_copies
                if voted_class_idx == target_label_int:
                    st.progress(pct, text=f"✅ {voted_name}: {count} votes")
                else:
                    st.progress(pct, text=f"❌ {voted_name}: {count} votes")
                    
        if show_heatmap: st.image(overlay_h, caption="Robust Attention Map", use_container_width=True)

# --- LOSS LANDSCAPE TAB ---
with tab_landscape:
    st.markdown("### Decision Boundary Visualization")
    st.markdown("Generates a 3D loss surface plot by perturbing the current image along two random normalized directions. Wide, flat valleys indicate robust models.")
    
    if st.button("Generate Loss Landscape for Current Image"):
        with st.spinner("Calculating loss surface (this takes ~30 seconds)..."):
            img_tensor, label_int = val_dataset[img_idx]
            img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
            label_tensor = torch.tensor([label_int], dtype=torch.long).to(DEVICE)
            
            # Use appropriate grid steps for speed vs quality
            landscape_path = visualize_loss_landscape(hero, DEVICE, img_tensor, label_tensor, epsilon=0.1, steps=20)
            if os.path.exists(landscape_path):
                st.image(landscape_path, caption="TRADES ResNet18 Loss Landscape", use_container_width=True)
            else:
                st.error("Failed to generate landscape plot.")

# --- PATCH ATTACKS TAB ---
with tab_patch:
    st.markdown("### Universal Adversarial Patch Analysis")
    st.markdown("Evaluates vulnerability to localized, high-confidence physical-world patches vs. standard Lp-norm perturbations.")
    
    patch_path = os.path.join(config.DATA_DIR, "patch_attack", "universal_patch.pt")
    
    if os.path.exists(patch_path):
        patch_tensor = torch.load(patch_path, map_location=DEVICE)
        st.image(to_display(patch_tensor), caption="Loaded Universal Patch", width=150)
        
        if st.button("Apply Patch to Current Image"):
            with st.spinner("Injecting patch and recalculating saliency..."):
                class FixedPatch(torch.nn.Module):
                    def __init__(self, t): super().__init__(); self.patch = t
                    def forward(self): return self.patch
                
                applier = PatchApplier(DEVICE, img_size=64, min_scale=0.3, max_scale=0.4)
                img_tensor, label_int = val_dataset[img_idx]
                img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
                label_tensor = torch.tensor([label_int], dtype=torch.long).to(DEVICE)
                
                # Original
                orig_pred = victim(img_tensor).argmax(1).item()
                orig_saliency = get_saliency_map(victim, img_tensor, orig_pred)
                
                # Patched
                victim.zero_grad()
                img_tensor.requires_grad = False
                patched_img = applier(img_tensor, FixedPatch(patch_tensor).patch)
                patched_pred = victim(patched_img).argmax(1).item()
                patched_saliency = get_saliency_map(victim, patched_img.detach(), patched_pred)
                
                orig_pred_name = class_mapping.get(orig_pred, f"Class {orig_pred}")
                patched_pred_name = class_mapping.get(patched_pred, f"Class {patched_pred}")
                
                # Display
                p1, p2, p3, p4 = st.columns(4)
                with p1: 
                    st.image(to_display(img_tensor), caption=f"Orig Pred: {orig_pred_name}", use_container_width=True)
                with p2:
                    st.image(orig_saliency.cpu().squeeze().numpy(), caption="Orig Saliency", clamp=True, use_container_width=True)
                with p3:
                    st.image(to_display(patched_img), caption=f"Patched Pred: {patched_pred_name}", use_container_width=True)
                with p4:
                    st.image(patched_saliency.cpu().squeeze().numpy(), caption="Patched Saliency", clamp=True, use_container_width=True)
    else:
        st.warning(f"Patch file not found at {patch_path}. Need to train one first!")

st.info("Note: Models trained on Tiny ImageNet (200 classes).")
