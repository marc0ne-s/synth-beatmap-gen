"""
The "Generative Ghost" Probe.
Validates the early convergence of the Transformer Decoder by peeking into its output 
for a specific test map and generating a human-readable visual matrix of X,Y positions.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Fix paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.transformer import TransformerCausalDecoder
from src.data.streaming_loader import DIFFICULTY_MAP

def infer_and_print():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n[Ghost Probe] Booting on {device}...")

    # Load Epoch 5 Checkpoint (The Silent Burn Finale)
    ckpt_path = "models/checkpoints/transformer_pilot_ep5.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "models/checkpoints/transformer_pilot_ep4.pt"
        if not os.path.exists(ckpt_path):
            print("No checkpoints found. Waiting for training...")
            return

    model = TransformerCausalDecoder(d_model=256, num_layers=4, d_audio=128, d_target=6).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"[Ghost Probe] Loaded memory state: {ckpt_path}")

    # Pick a random "Hard" map that has both audio and features
    features_dir = Path("dataset/features")
    hard_maps = list(features_dir.glob("*_Hard.npz"))
    
    if not hard_maps:
        print("No Hard maps found.")
        return
        
    test_map = np.random.choice(hard_maps)
    stem = test_map.stem
    uuid = stem.rsplit("_", 1)[0]
    
    audio_path = Path("dataset/audio_features") / f"{uuid}.npz"
    if not audio_path.exists():
        print(f"Skipping {uuid}: No audio available.")
        return

    print(f"[Ghost Probe] Probing Test Map: {uuid} (Hard)\n")

    # Load actual input parameters exactly as data loader would
    try:
        audio_data = np.load(audio_path)["audio_mel"]
        feat_data = np.load(test_map)
        pos = torch.from_numpy(feat_data["note_positions"]).float()
        pres = torch.from_numpy(feat_data["note_presence"]).float()
    except Exception as e:
        print(f"Failed to load map arrays: {e}")
        return

    # Tanh normalization matching our stabilization patch
    T = min(pos.shape[0], audio_data.shape[0], 4096)
    audio_t = torch.from_numpy(audio_data[:T]).float().unsqueeze(0).to(device)
    
    flat_pos = pos[:T].view(T, 4)
    flat_pos = torch.tanh(flat_pos / 3.0)
    
    tgt_t = torch.cat([pres[:T], flat_pos], dim=-1).unsqueeze(0).to(device)
    
    diff_idx = torch.tensor([DIFFICULTY_MAP["Hard"]], dtype=torch.long).to(device)
    
    # -------------------------------------------------------------
    # Forward Pass Autoregressively / Causally
    # -------------------------------------------------------------
    with torch.no_grad():
        with torch.amp.autocast(device_type="mps", dtype=torch.bfloat16) if device.type == "mps" else __import__('contextlib').nullcontext():
            # Since our model is sequence-to-sequence causal, passing the whole
            # track outputs predictions for all t based on <t.
            preds = model(audio_t, tgt_t, diff_idx)
            
    # Squeeze out batch dimension
    pres_logits = preds["presence_logits"].squeeze(0)  # (T, 2)
    pos_preds = preds["position_pred"].squeeze(0)      # (T, 4)
    
    # Sigmoid for probabilities
    pres_probs = torch.sigmoid(pres_logits)

    # De-normalize tanh
    # If output = tanh(X / 3), then X = 3 * arctanh(output). 
    pos_dec = torch.arctanh(torch.clamp(pos_preds, -0.99, 0.99)) * 3.0
    
    # Let's inspect 3 specific time zones (Start, Mid, End)
    zones = {
        "THE BUILDUP (Starts ~10% in)": int(T * 0.1),
        "THE DROP (Middle ~50% in)": int(T * 0.5),
        "THE CLIMAX (Near End ~90% in)": int(T * 0.9),
    }

    for zone_name, frame in zones.items():
        print(f"=== {zone_name} | Frame {frame} ===")
        # Look at a short 10-frame window (approx 200ms)
        printed = False
        for t in range(frame, frame + 25):  # Look up to 25 frames
            if t >= T: break
            
            p_right, p_left = pres_probs[t, 0].item(), pres_probs[t, 1].item()
            rx, ry = pos_dec[t, 0].item(), pos_dec[t, 1].item()
            lx, ly = pos_dec[t, 2].item(), pos_dec[t, 3].item()
            
            # Formatting the printout
            right_logic = f"RIGHT [X:{rx:5.2f} Y:{ry:5.2f}]" if p_right > 0.4 else "     -     "
            left_logic = f"LEFT  [X:{lx:5.2f} Y:{ly:5.2f}]" if p_left > 0.4 else "     -     "
            
            # Ground truth for comparison
            gt_pr, gt_pl = pres[t, 0].item(), pres[t, 1].item()
            
            # Only print frames where there is actual activity predicted or true
            if p_right > 0.3 or p_left > 0.3 or gt_pr > 0.5 or gt_pl > 0.5:
                print(f" T+{t:4d} | PRED: {right_logic} | {left_logic}  ||  GROUND TRUTH: {'R' if gt_pr>0.5 else '-'} {'L' if gt_pl>0.5 else '-'}")
                printed = True
        if not printed:
            print("  (No note presence detected in this window)")

    print("\n[Ghost Probe] Analysis Complete.")

if __name__ == "__main__":
    infer_and_print()
