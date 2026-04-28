#!/usr/bin/env python3
"""Update dashboard status.json with live system data."""
import json, os, subprocess, glob, time
from pathlib import Path

CKPT_DIR = Path("/Volumes/Second-Brain-1/AI/Synth/models/checkpoints")
OUT = Path("/Volumes/Second-Brain-1/AI/Synth/dashboard/status.json")

def get_training_state():
    try:
        ps = subprocess.check_output(
            ["ps", "-p", "34021", "-o", "etime,pcpu,pmem"],
            text=True, timeout=2
        ).strip().split("\n")[1].strip()
        parts = ps.split()
        return {"running": True, "etime": parts[0], "cpu": parts[1], "mem": parts[2]}
    except:
        return {"running": False, "etime": "--", "cpu": "0", "mem": "0"}

def get_checkpoints():
    pts = sorted(CKPT_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in pts[:5]]

def estimate_epoch():
    # Phase 12 started ~1 hour ago, ~1hr per epoch
    try:
        started = 1745734980  # approximate start time
        elapsed_hours = (time.time() - started) / 3600
        epoch = min(int(elapsed_hours) + 1, 10)
        return epoch
    except:
        return 1

state = {
    "timestamp": int(time.time()),
    "training": get_training_state(),
    "checkpoints": get_checkpoints(),
    "epoch": {
        "current": estimate_epoch(),
        "total": 10,
        "batch": "~20",
        "loss": 81.3
    },
    "metrics": {
        "phase10": {"recall": 53.61, "precision": 37.88, "mse": 19.82},
        "phase11": {"recall": 41.30, "precision": 40.63, "mse": 13.57},
        "phase12": {"recall": None, "precision": None, "mse": None},
        "target": {"recall": 85.0, "precision": 80.0, "mse": 2.0}
    }
}

OUT.write_text(json.dumps(state, indent=2))
