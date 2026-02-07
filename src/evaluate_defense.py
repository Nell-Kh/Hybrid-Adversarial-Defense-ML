import os
import subprocess
import config

def main():
    print("="*40)
    print("   HYBRID DEFENSE EVALUATION SYSTEM   ")
    print("="*40)
    
    # 1. Base Model Check
    print("\n[Stage 1] Checking Target Model...")
    subprocess.run(["python3", "src/evaluate.py"])
    
    # 2. Attack Generation
    print("\n[Stage 2] Generating Adversarial Attacks (AutoAttack)...")
    # This script generates the samples that both detectors will test
    subprocess.run(["python3", "src/auto_attack.py"])
    
    # 3. Detector 1: Stability (The "Physics" Approach)
    print("\n[Stage 3] Testing Detector 1: Prediction Stability...")
    subprocess.run(["python3", "src/detectors.py"])

    # 4. Detector 2: Deep Features (The "Abstract" Approach)
    print("\n[Stage 4] Testing Detector 2: Mahalanobis Distance...")
    subprocess.run(["python3", "src/mahalanobis_detector.py"])
    
    print("\n" + "="*40)
    print("   EVALUATION COMPLETE   ")
    print("="*40)

if __name__ == "__main__":
    main()
