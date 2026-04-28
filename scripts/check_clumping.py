import os
import json
import numpy as np
from pathlib import Path

def check_clumping(report_dir):
    reports = list(Path(report_dir).glob("*.json"))
    stats = []
    
    for rp in reports:
        with open(rp, "r") as f:
            data = json.load(f)
        
        for diff, results in data.get("difficulties", {}).items():
            # Get notes from the gold bundle for this UUID if possible,
            # but wait, let's just look at the report's density ramp.
            ramp = results.get("density_ramp", {})
            early_nps = ramp.get("early_nps", 0)
            late_nps = ramp.get("late_nps", 0)
            ratio = ramp.get("ramp_ratio", 1.0)
            
            # Simple Clumping Check: If one half has 3x more NPS than the other
            if ratio > 3.0 or ratio < 0.33:
                stats.append({"uuid": rp.stem, "diff": diff, "ratio": ratio})
                
    return stats

if __name__ == "__main__":
    report_dir = "/Volumes/Second-Brain-1/AI/Synth/evaluation/phase12b/audit_ep5_reports/"
    clumps = check_clumping(report_dir)
    print(f"Detected {len(clumps)} clumped maps in audit.")
    for c in clumps[:5]:
        print(c)
