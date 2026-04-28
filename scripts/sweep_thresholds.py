#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

def run_sweep(ckpt_path):
    thresholds = [0.20, 0.30, 0.40, 0.50, 0.70]
    
    print(f"=====================================================")
    print(f"  THRESHOLD SWEEP: {Path(ckpt_path).name}")
    print("=====================================================")
    
    script_path = Path("/Volumes/Second-Brain-1/AI/Synth/scripts/evaluate_phase12.py")
    
    for t in thresholds:
        print(f"\n[+] Testing Threshold: {t}")
        cmd = [
            "python3", str(script_path),
            "--threshold", str(t),
            "--checkpoint", ckpt_path
        ]
        subprocess.run(cmd)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 sweep_thresholds.py <path_to_checkpoint>")
        sys.exit(1)
        
    run_sweep(sys.argv[1])
