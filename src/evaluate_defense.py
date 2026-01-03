import os
import subprocess
import config

def main():
    print("="*40)
    print("   ADVANCED DEFENSE EVALUATION   ")
    print("="*40)
    
    # 1. Base Model Check
    print("\n[Stage 1] Checking Base Model...")
    # Using subprocess to ensure clean memory
    subprocess.run(["python3", "src/evaluate.py"])
    
    # 2. Attack Generation (Implicitly covered in detectors.py now, but good to have explicit check)
    print("\n[Stage 2] Attack Module Check...")
    # Just printing info, the detectors run the attacks internally now to ensure fairness
    print("Using AutoAttackLite (PGD-CE + PGD-DLR)")
    
    # 3. Defense Evaluation
    print("\n[Stage 3] Testing Stability Detector (Physics Mode)...")
    subprocess.run(["python3", "src/detectors.py"])
    
    print("\n" + "="*40)
    print("   EVALUATION COMPLETE   ")
    print("="*40)

if __name__ == "__main__":
    main()
