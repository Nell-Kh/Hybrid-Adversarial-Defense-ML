import os
import subprocess
import config

def main():
    print("="*40)
    print("   ADVANCED DEFENSE EVALUATION   ")
    print("="*40)
    
    # 1. Base Model Check
    print("\n[Stage 1] Checking Base Model...")
    from evaluate import evaluate
    evaluate()
    
    # 2. Attack Generation
    print("\n[Stage 2] Generating Advanced Attacks (AutoAttack)...")
    # We call the script directly
    subprocess.run(["python3", "src/auto_attack.py"])
    
    # 3. Defense Evaluation
    print("\n[Stage 3] Testing Mahalanobis Detector...")
    subprocess.run(["python3", "src/mahalanobis_detector.py"])
    
    print("\n" + "="*40)
    print("   EVALUATION COMPLETE   ")
    print("="*40)

if __name__ == "__main__":
    main()
