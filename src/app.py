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
from attacks.eot import EoTAttacker
from defenses.mahalanobis import MahalanobisDetector
from defenses.stochastic_ensemble import TTA_Ensemble
from defenses.image_cleaning import apply_cleaning
from defenses.certified_robustness import CertifiedRobustness
from utils_vis import GradCAM, apply_heatmap
from visualize_landscape import visualize_loss_landscape
from visualize_patch import get_saliency_map

# --- CONFIG ---
st.set_page_config(page_title="Adversarial Robustness Evaluation", layout="wide")
DEVICE = config.DEVICE

def compute_stealth_metrics(clean_tensor, adv_tensor):
    import numpy as np
    from skimage.metrics import structural_similarity as ssim
    
    with torch.no_grad():
        diff = adv_tensor - clean_tensor
        l2 = torch.norm(diff.view(diff.shape[0], -1), p=2, dim=1).mean().item()
        linf = torch.norm(diff.view(diff.shape[0], -1), p=float('inf'), dim=1).mean().item()
        
        c_np = (clean_tensor.squeeze().cpu().permute(1, 2, 0).numpy() + 1.0) / 2.0
        a_np = (adv_tensor.squeeze().cpu().permute(1, 2, 0).numpy() + 1.0) / 2.0
        c_np = np.clip(c_np, 0, 1)
        a_np = np.clip(a_np, 0, 1)
        
        mse = np.mean((c_np - a_np) ** 2)
        psnr = float('inf') if mse == 0 else 20 * np.log10(1.0) - 10 * np.log10(mse)
        
        try:
            ssim_val = ssim(c_np, a_np, data_range=1.0, channel_axis=-1)
        except Exception:
            ssim_val = 0.0
            
    return l2, linf, psnr, ssim_val

# --- CACHED LOADERS ---
@st.cache_resource
def load_models():
    # 1. Standard Model (standard)
    Standard = get_model(DEVICE)
    Standard_path = os.path.join("models", "resnet_tinyimagenet.pth")
    if os.path.exists(Standard_path):
        Standard.load_state_dict(torch.load(Standard_path, map_location=DEVICE))
    else:
        st.error(f"Standard model not found at {Standard_path}")
    Standard.eval()
    
    # 2. Robust Model (Defense)
    Robust = get_model(DEVICE)
    Robust_path = os.path.join("models", "resnet_robust.pth")
    if os.path.exists(Robust_path):
        checkpoint = torch.load(Robust_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            Robust.load_state_dict(checkpoint['model_state_dict'])
        else:
            Robust.load_state_dict(checkpoint)
    else:
        st.warning(f"Robust model not found at {Robust_path}. Falling back to standard model for robust predictions.")
        # Fallback so the app doesn't crash completely
        Robust = Standard
    Robust.eval()
    
    # 3. Mahalanobis Detector (Mahalanobis)
    detector = MahalanobisDetector(Standard, DEVICE)
    detector.load_stats() # Attempts to load from disk
    
    # 4. Stochastic Ensemble (TTA / Randomized Smoothing)
    # Wraps the robust model to provide Test-Time Augmentation defenses
    tta_Robust = TTA_Ensemble(Robust, num_copies=10, max_shift=2, noise_std=0.02)
    
    # 5. Certified Robustness Evaluator (Mathematical Guarantees)
    # n=50 for speed in UI, normally n=10,000 for academic papers
    certifier = CertifiedRobustness(Robust, DEVICE, noise_std=0.1, n0=10, n=50)
    
    return Standard, Robust, tta_Robust, detector, certifier

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

with st.expander("Overview: Adversarial Machine Learning Framework"):
    st.markdown("""
    **Adversarial Machine Learning Dashboard**
    
    This platform evaluates the vulnerability of standard convolutional neural networks to adversarial noise, 
    and tests the efficacy of various mathematical defense mechanisms.
    
    *   **Attacks**: Apply calculated perturbations to force misclassification.
    *   **Defenses**: Evaluate robust training paradigms and statistical detection methods.
    
    Use the controls on the left to select an evaluation sample, configure an attack trajectory, and run the pipeline.
    """)

st.markdown("### Comparative Analysis: Standard ResNet18 vs. TRADES ResNet18")

# Create Tabs
tab_eval, tab_landscape, tab_patch, tab_radar = st.tabs(["Evaluation & Defense", "Loss Landscape", "Patch Attacks", "Radar Benchmark"])

Standard, Robust, tta_Robust, detector, certifier = load_models()
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
        "Boundary (Black-Box)",
        "EoT Oracle (Adaptive)"
    ])
    
    # Mathematical Tooltips
    attack_explanations = {
        "FGSM (Fast Gradient)": "**FGSM:** Computes the gradient of the loss function with respect to the input pixels and adds a scaled perturbation in the direction of the sign.\n\n$x_{adv} = x + \epsilon \cdot sign(\\nabla_x J(x, y))$",
        "DeepFool (Minimum Norm)": "**DeepFool:** An iterative L2 attack that projects the input onto the closest linear approximation of the decision boundary, seeking the minimal possible perturbation.",
        "AutoAttack (Ensemble)": "**AutoAttack:** A parameter-free ensemble of established attacks (APGD-ce, APGD-dlr, FAB, Square) used to evaluate worst-case robustness.",
        "C&W (L2 Optimization)": "**Carlini & Wagner (C&W):** Formulates the attack as an optimization problem, minimizing $||\delta||_2^2 + c \cdot f(x+\delta)$ subject to box constraints.",
        "Ninja (Adaptive PGD)": "**Adaptive PGD:** A defense-aware attack that modifies the PGD objective function to maximize classification error while minimizing the detector's anomaly score.",
        "Boundary (Black-Box)": "**Boundary Attack:** A decision-based black-box method that initializes from a completely different class and performs a random walk along the decision boundary.",
        "EoT Oracle (Adaptive)": "**Expectation over Transformation (EoT):** Designed to bypass stochastic defenses by calculating expected gradients across a distribution of transformations."
    }
    
    if attack_name != "None":
        st.info(attack_explanations[attack_name])
    
    # Dynamic parameters
    epsilon = st.slider("Perturbation Magnitude (Epsilon)", 0.0, 0.1, 0.031, step=0.001, help="Controls how much noise the attacker is allowed to add. Higher = more visible noise but stronger attack.")
    
    if attack_name in ["AutoAttack (Ensemble)", "C&W (L2 Optimization)", "Ninja (Adaptive PGD)", "Boundary (Black-Box)", "EoT Oracle (Adaptive)"]:
        steps = st.slider("Optimization Steps", 10, 200, 50, step=10)
        
    st.divider()
    st.header("Attack Objectives")
    enable_targeting = st.checkbox("Enable Targeted Spoofing (C&W and Ninja only)", value=False, help="Force the attack to make the model predict a specific incorrect label instead of just ANY incorrect label.")
    
    target_class_idx = None
    if enable_targeting and attack_name in ["C&W (L2 Optimization)", "Ninja (Adaptive PGD)"]:
        name_to_label = {v: k for k, v in class_mapping.items()}
        target_class_name = st.selectbox("Spoof Target Class", sorted_class_names, help="Pick the class you want the model to hallucinate.")
        target_class_idx = name_to_label[target_class_name]
    elif enable_targeting:
        st.warning(f"Targeting is not currently supported for {attack_name}.")
    
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
                output = Standard(target_image)
                loss = nn.CrossEntropyLoss()(output, target_label)
                Standard.zero_grad()
                loss.backward()
                data_grad = target_image.grad.data
                sign_data_grad = data_grad.sign()
                adv_image = target_image + epsilon * sign_data_grad
                adv_image = torch.clamp(adv_image, -1, 1)
                target_image.requires_grad = False
                
            elif attack_name == "DeepFool (Minimum Norm)":
                attacker = DeepFool(Standard, DEVICE, overshoot=0.02, max_iters=steps if 'steps' in locals() else 50)
                adv_image = attacker.attack(target_image, target_label)
                
            elif attack_name == "AutoAttack (Ensemble)":
                attacker = AutoAttackLite(Standard, DEVICE, eps=epsilon)
                adv_image = attacker.attack(target_image, target_label)
                
            elif attack_name == "C&W (L2 Optimization)":
                attacker = CWAttacker(Standard, DEVICE, steps=steps if 'steps' in locals() else 50)
                target_tensor = torch.tensor([target_class_idx], dtype=torch.long).to(DEVICE) if target_class_idx is not None else None
                adv_image = attacker.attack(target_image, target_label, target_labels=target_tensor)
                
            elif attack_name == "Ninja (Adaptive PGD)":
                attacker = AdaptiveAttacker(Standard, DEVICE, detector, eps=epsilon, steps=steps if 'steps' in locals() else 50)
                target_tensor = torch.tensor([target_class_idx], dtype=torch.long).to(DEVICE) if target_class_idx is not None else None
                adv_image = attacker.attack(target_image, target_label, target_labels=target_tensor)
                
            elif attack_name == "Boundary (Black-Box)":
                attacker = BoundaryAttack(Standard, DEVICE, steps=steps if 'steps' in locals() else 50)
                adv_image = attacker.attack(target_image, target_label)
                
            elif attack_name == "EoT Oracle (Adaptive)":
                st.warning("The Oracle is actively simulating realities to defeat the Stochastic Ensemble...")
                # We attack the 'Robust' (robust model) since EoT is designed to defeat its defenses
                attacker = EoTAttacker(Robust, DEVICE, eps=epsilon, steps=steps if 'steps' in locals() else 20, eot_samples=10, max_shift=2)
                adv_image = attacker.attack(target_image, target_label)
        
        noise = (adv_image - target_image).abs()
                
    # --- RAW INFERENCE (To prove the attack worked before cleaning) ---
    with torch.no_grad():
        raw_pred_Standard_logits = Standard(adv_image)
        raw_pred_Standard = raw_pred_Standard_logits.argmax(1).item()
        
    # --- DISPLAY ---
    with torch.no_grad():
        pred_Standard_logits = Standard(adv_image)
        pred_Standard = pred_Standard_logits.argmax(1).item()
        
        # Branch defense evaluation based on UI Toggle
        if enable_tta:
            # We pass return_consensus=True to get the dictionary of votes
            pred_Robust_logits, vote_breakdown = tta_Robust(adv_image, return_consensus=True)
            pred_Robust = pred_Robust_logits.argmax(1).item()
        else:
            pred_Robust_logits = Robust(adv_image)
            pred_Robust = pred_Robust_logits.argmax(1).item()
            vote_breakdown = None
            
        # Calculate Certified Robustness
        if run_analysis:
            with st.spinner("Calculating Certified Smoothing Radius..."):
                cert_class, cert_radius, cert_prob = certifier.certify(adv_image)
        else:
            cert_class, cert_radius, cert_prob = -1, 0.0, 0.0
        
        # Calculate Trust Score
        try:
            trust_score_val = detector.get_trust_score(adv_image)[0] * 100  # Percentage
        except Exception as e:
            trust_score_val = 100.0 # Default to trusted if detector fails
    
    vis_Standard = adv_image
    vis_Robust = adv_image
    
    if show_heatmap:
        cam_v = GradCAM(Standard, Standard.layer4[-1])
        heatmap_v = cam_v(adv_image, pred_Standard)
        overlay_v = apply_heatmap(adv_image * 0.5 + 0.5, heatmap_v)
        cam_h = GradCAM(Robust, Robust.layer4[-1])
        heatmap_h = cam_h(adv_image, pred_Robust)
        overlay_h = apply_heatmap(adv_image * 0.5 + 0.5, heatmap_h)
    
    # Render Stats Summary
    st.sidebar.divider()
    st.sidebar.subheader("Mahalanobis Distance Detector")
    st.sidebar.markdown("Evaluates internal feature representations to detect out-of-distribution adversarial samples.")
    st.sidebar.progress(int(trust_score_val))
    if trust_score_val < 50:
        st.sidebar.warning(f"Anomaly Detected (Trust: {trust_score_val:.1f}%)")
    else:
        st.sidebar.success(f"Sample In-Distribution (Trust: {trust_score_val:.1f}%)")

    # Pre-process noise for visualization (min-max normalization to make patterns visible)
    display_noise = noise.detach().squeeze().cpu().permute(1,2,0).numpy()
    noise_max = display_noise.max()
    if noise_max > 0:
        display_noise = display_noise / noise_max

    # --- SECTION 1: THE ATTACK VECTOR (3 Symmetrical Columns) ---
    st.subheader("1. Input Perturbation Vector", help="Visualization of the applied adversarial noise delta.")
    c1, c2, c3 = st.columns(3)
    with c1: 
        st.image(to_display(target_image), caption=f"Original Sample ({class_name})", use_container_width=True)
    with c2: 
        st.image(display_noise, caption="Adversarial Delta (Normalized)", clamp=True, use_container_width=True)
    with c3: 
        st.image(to_display(adv_image), caption="Perturbed Sample", use_container_width=True)
        
    # --- ACADEMIC STEALTH METRICS ---
    if run_analysis and attack_name != "None":
        st.markdown("**Academic Stealth Metrics (Imperceptibility)**")
        m1, m2, m3, m4 = st.columns(4)
        l2_val, linf_val, psnr_val, ssim_val = compute_stealth_metrics(target_image, adv_image)
        
        m1.metric("L2 Norm (Dist)", f"{l2_val:.3f}", help="Total Euclidean noise distance. Lower is better.")
        m2.metric("L-inf Norm (Max)", f"{linf_val:.3f}", help="Maximum change to any single pixel. Lower is better.")
        m3.metric("PSNR (Signal/Noise)", f"{psnr_val:.1f} dB", help="Peak Signal-to-Noise Ratio. Higher is better (>30dB is usually imperceptible).")
        m4.metric("SSIM (Similarity)", f"{ssim_val:.3f}", help="Structural Similarity Index. 1.0 means perfectly identical to human eye.")
        
    st.divider()
    
    # --- SECTION 2: COMPARATIVE INFERENCE (2 Symmetrical Columns) ---
    st.subheader("2. Model Inference Comparison", help="Comparing how the standard standard model reacts to the attack vs. the fortified TRADES model.")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Standard ResNet18 (standard)", help="A standard, unprotected AI model. It performs very well on normal images but is highly vulnerable to being tricked by attacks.")
        
        pred_vic_name = class_mapping.get(pred_Standard, f"Class {pred_Standard}")
        if pred_Standard == target_label_int: st.success(f"Prediction: CORRECT ({pred_vic_name})")
        else: st.error(f"Prediction: INCORRECT ({pred_vic_name})")
        
        if show_heatmap: st.image(overlay_v, caption="standard Attention Map", use_container_width=True)
            
    with col_b:
        st.subheader("TRADES ResNet18 (Robust)", help="A fortified AI model defended via TRADES (Tradeoff-inspired Adversarial Defense). It fundamentally restructures its own neural pathways during training to resist malicious adversarial vectors.")
        pred_Robust_name = class_mapping.get(pred_Robust, f"Class {pred_Robust}")
        if pred_Robust == target_label_int: st.success(f"Prediction: CORRECT ({pred_Robust_name})")
        else: st.warning(f"Prediction: INCORRECT ({pred_Robust_name})")
        
        if show_heatmap: 
            st.image(overlay_h, caption="Robust Attention Map (Grad-CAM)", use_container_width=True)
            
        if enable_tta and vote_breakdown:
            st.markdown("---")
            st.markdown("**Stochastic Ensemble Consensus Vote Tracker:**")
            for voted_class_idx, count in vote_breakdown.items():
                voted_name = class_mapping.get(voted_class_idx, f"Class {voted_class_idx}")
                pct = count / tta_Robust.num_copies
                if voted_class_idx == target_label_int:
                    st.progress(pct, text=f"[CORRECT] {voted_name}: {count} votes")
                else:
                    st.progress(pct, text=f"[INCORRECT] {voted_name}: {count} votes")
        
    st.divider()
    
    # --- SECTION 3: AUTONOMOUS PURIFICATION MATRIX ---
    st.subheader("3. Autonomous Purification Matrix", help="Silently testing the adversarial image against multiple image-processing and neural defenses simultaneously.")
    
    if run_analysis and attack_name != "None":
        st.markdown("**How it works:** We apply each filtering defense to the adversarial image, and then feed the cleaned image back into the **Standard ResNet18 (standard)**. This proves whether the defense actually successfully scrubbed the attack noise from the image. Notice how basic filters often fail, while Neural ML succeeds.")
        
        filters = {
            "Gaussian Blur": "Gaussian Blur",
            "Bit-Depth (4-bit)": "Bit Depth Reduction (4-bit)",
            "Median Filter": "Median Filter",
            "FFT Low-Pass": "FFT Low-Pass Filter",
            "Deep Autoencoder": "Neural Denoising Autoencoder (DAE)"
        }
        
        # Create dynamic columns for each filter
        filter_cols = st.columns(len(filters))
        
        for idx, (display_name, method_name) in enumerate(filters.items()):
            with filter_cols[idx]:
                st.markdown(f"**{display_name}**")
                
                # Handle untrained Autoencoder gracefully
                if method_name == "Neural Denoising Autoencoder (DAE)" and not os.path.exists("models/neural_cleaner.pth"):
                    st.warning("Untrained")
                    st.caption("Run training script.")
                    st.image(np.zeros((64, 64, 3)), caption="No weights found", use_container_width=True)
                    continue
                
                # Apply filter
                cleaned_tensor = apply_cleaning(adv_image, method_name, DEVICE)
                
                # Get standard prediction on cleaned image
                with torch.no_grad():
                    clean_logits = Standard(cleaned_tensor)
                    clean_pred = clean_logits.argmax(1).item()
                    clean_conf = torch.softmax(clean_logits, dim=1)[0, target_label_int].item() * 100
                    
                clean_name = class_mapping.get(clean_pred, f"Class {clean_pred}")
                
                if clean_pred == target_label_int:
                    st.success("✅ Defeated")
                    st.caption(f"Conf: {clean_conf:.1f}%")
                else:
                    st.error("❌ Persists")
                    st.caption(f"Pred: {clean_name}")
                
                # Display a tiny thumbnail of the cleaned image
                st.image(to_display(cleaned_tensor), use_container_width=True)
    else:
        st.info("Run an attack to evaluate the Autonomous Purification Matrix.")
        
    st.divider()

    # --- SECTION 4: DEFENSIVE ANALYTICS DASHBOARD ---
    st.subheader("4. Live Defensive Analytics Dashboard", help="Real-time mathematical breakdown of model confidence and certified robustness guarantees.")
    
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.markdown("**1. Standard Model Vulnerability**", help="How confident the unprotected AI is in the attacker's forced hallucination.")
        
        if enable_targeting and target_class_idx is not None:
            eval_class_idx = target_class_idx
            eval_label = "Target Class Confidence"
        else:
            eval_class_idx = pred_Standard
            eval_label = "Adversarial Confidence"
            
        standard_pct = torch.softmax(pred_Standard_logits, dim=1)[0, eval_class_idx].item() * 100
        st.progress(int(standard_pct), text=f"{eval_label}: {standard_pct:.1f}%")
        
    with metric_col2:
        st.markdown("**2. TRADES Suppression**", help="How well the Defended AI suppresses the attacker's forced hallucination.")
        robust_pct = torch.softmax(pred_Robust_logits, dim=1)[0, eval_class_idx].item() * 100
        st.progress(int(robust_pct), text=f"{eval_label}: {robust_pct:.1f}%")
        
    with metric_col3:
        st.markdown("**3. Certified Mathematical Radius**", help="The Holy Grail of AI defense. This uses 'Randomized Smoothing' to mathematically PROVE that absolutely NO ATTACK with an intensity lower than 'Radius (R)' can ever trick the model. It is a 100% guarantee.")
        if cert_radius > 0.0:
            st.success(f"**Radius (R)**: {cert_radius:.3f} | **pA**: {cert_prob:.2f}")
            st.caption(f"Guaranteed Safe against any attack with L2 norm < {cert_radius:.3f}.")
        else:
            st.error("Model too uncertain to guarantee safety.")
            st.caption("No mathematical guarantee exists for this input.")

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
            fig = visualize_loss_landscape(Robust, DEVICE, img_tensor, label_tensor, epsilon=0.1, steps=20)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Failed to generate interactive landscape plot.")

# --- PATCH ATTACKS TAB ---
with tab_patch:
    st.markdown("### Universal Adversarial Patch Analysis")
    st.markdown("Evaluates vulnerability to localized, high-confidence physical-world patches vs. standard Lp-norm perturbations.")
    
    patch_path = os.path.join(config.DATA_DIR, "patch_attack", "universal_patch.pt")
    
    if os.path.exists(patch_path):
        patch_tensor = torch.load(patch_path, map_location=DEVICE)
        st.image(to_display(patch_tensor), caption="Loaded Universal Patch", width=150)
        img_tensor, label_int = val_dataset[img_idx]
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        label_tensor = torch.tensor([label_int], dtype=torch.long).to(DEVICE)
        
        st.markdown("### Interactive Patch Placement")
        st.markdown("Use the sliders to physically move the adversarial sticker around the image and see how localized vulnerabilities disrupt the model's Grad-CAM attention.")
        
        # Interactive Controls
        col1, col2, col3 = st.columns(3)
        with col1: patch_scale = st.slider("Sticker Scale", 0.1, 1.0, 0.35, step=0.05)
        
        target_size = int(64 * patch_scale)
        max_coord = 64 - target_size
        
        with col2: patch_x = st.slider("X Coordinate", 0, max(0, max_coord), int(max_coord/2))
        with col3: patch_y = st.slider("Y Coordinate", 0, max(0, max_coord), int(max_coord/2))
        
        # Calculate Original standard
        orig_pred = Standard(img_tensor).argmax(1).item()
        orig_saliency = get_saliency_map(Standard, img_tensor, orig_pred)
        orig_pred_name = class_mapping.get(orig_pred, f"Class {orig_pred}")
        
        # Apply Patch explicitly based on UI coordinates
        patched_img = img_tensor.clone()
        import torch.nn.functional as F
        patch_resized = F.interpolate(
            patch_tensor.unsqueeze(0), 
            size=(target_size, target_size), 
            mode='bilinear'
        ).squeeze(0)
        
        patched_img[0, :, patch_y:patch_y+target_size, patch_x:patch_x+target_size] = patch_resized
        patched_img = torch.clamp(patched_img, -1, 1)
        
        # Calculate Patched Prediction
        patched_pred = Standard(patched_img).argmax(1).item()
        patched_saliency = get_saliency_map(Standard, patched_img.detach(), patched_pred)
        patched_pred_name = class_mapping.get(patched_pred, f"Class {patched_pred}")
        
        # Display comparative tracking pipeline
        st.divider()
        st.subheader("Physical Attack Telemetry")
        
        p1, p2, p3, p4 = st.columns(4)
        with p1: 
            st.image(to_display(img_tensor), caption=f"Orig Pred: {orig_pred_name}", use_container_width=True)
            if orig_pred == label_int: st.success("Correct")
            else: st.error("Incorrect")
            
        with p2:
            st.image(orig_saliency.cpu().squeeze().numpy(), caption="Orig Saliency Focus", clamp=True, use_container_width=True)
            
        with p3:
            st.image(to_display(patched_img), caption=f"Patched Pred: {patched_pred_name}", use_container_width=True)
            if patched_pred == label_int: st.success("Correct")
            else: st.error("Attack Succeeded")
            
        with p4:
            st.image(patched_saliency.cpu().squeeze().numpy(), caption="Patched Saliency Hijack", clamp=True, use_container_width=True)
    else:
        st.warning(f"Patch file not found at {patch_path}. Need to train one first!")

# --- RADAR BENCHMARK TAB ---
with tab_radar:
    st.markdown("### Automated Evaluation Suite (Spider Chart)")
    st.markdown("Rapidly benchmarks both models against a gauntlet of attacks and visualizes their geometric robustness profiles.")
    
    if st.button("Run Radar Benchmark (takes ~45s)"):
        with st.spinner("Benchmarking models against a gauntlet of 10 random images across 4 attacks..."):
            import plotly.graph_objects as go
            import torch.utils.data
            from attacks.fgsm import FGSMAttacker
            
            # Grab 10 random images
            num_imgs = 10
            dl = torch.utils.data.DataLoader(val_dataset, batch_size=num_imgs, shuffle=True)
            images, labels = next(iter(dl))
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Define Attacker Gauntlet
            attacks = {
                "FGSM": FGSMAttacker(Standard, DEVICE, eps=8/255),
                "C&W": CWAttacker(Standard, DEVICE, steps=20),
                "Ninja (PGD)": AdaptiveAttacker(Standard, DEVICE, detector, eps=8/255, steps=10),
                "AutoAttack": AutoAttackLite(Standard, DEVICE, eps=8/255)
            }
            
            results_Standard = {}
            results_Robust = {}
            
            # Calculate Clean Accuracy Focus
            with torch.no_grad():
                results_Standard["Clean Accuracy"] = (Standard(images).argmax(1) == labels).float().mean().item() * 100
                results_Robust["Clean Accuracy"] = (Robust(images).argmax(1) == labels).float().mean().item() * 100
            
            # Execute Gauntlet
            for atk_name, attacker in attacks.items():
                adv_imgs = attacker.attack(images, labels)
                with torch.no_grad():
                    results_Standard[atk_name] = (Standard(adv_imgs).argmax(1) == labels).float().mean().item() * 100
                    results_Robust[atk_name] = (Robust(adv_imgs).argmax(1) == labels).float().mean().item() * 100
                
            categories = list(results_Standard.keys())
            categories_loop = categories + [categories[0]]
            
            vic_vals = list(results_Standard.values())
            vic_vals.append(vic_vals[0])
            
            Robust_vals = list(results_Robust.values())
            Robust_vals.append(Robust_vals[0])
            
            # Build Radar
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=vic_vals, theta=categories_loop, fill='toself', name='standard (Standard)', line_color='red'))
            fig.add_trace(go.Scatterpolar(r=Robust_vals, theta=categories_loop, fill='toself', name='TRADES (Robust)', line_color='green'))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                title="Aggregate Robustness Profile (Accuracy %)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.success("Evaluation complete. Radar chart indicates standard model degradation under perturbation vs TRADES stability.")
            


st.info("Note: Models trained on Tiny ImageNet (200 classes).")
