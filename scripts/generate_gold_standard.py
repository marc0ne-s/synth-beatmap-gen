#!/usr/bin/env python3
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path("/Volumes/Second-Brain-1/AI/Synth")))

from src.models.transformer import TransformerCausalDecoder

# === PRODUCTION PARAMETERS ===
NMS_WINDOW = 100.0  # ms (The "Nyquist" Sweet Spot for v12b)
VERSION_TAG = "AIv12b.100ms"
DETECTION_THRESHOLD = 0.5
# =============================

def compute_augmented_features(mel: torch.Tensor) -> torch.Tensor:
    T, num_freq = mel.shape[1], mel.shape[2]
    mel_sq = mel[0]
    shifted_mel = torch.cat([torch.zeros(1, num_freq, device=mel.device), mel_sq[:-1, :]], dim=0)
    delta = mel_sq - shifted_mel
    delta_cpu = delta.cpu().unsqueeze(0)
    delta_44 = F.adaptive_avg_pool1d(delta_cpu, 44).squeeze(0).to(mel.device)
    flux = F.relu(delta).sum(dim=1, keepdim=True) / 10.0
    f_idx = torch.arange(num_freq, dtype=torch.float32, device=mel.device) / (num_freq - 1)
    mel_sum = mel_sq.sum(dim=1, keepdim=True) + 1e-8
    centroid = (mel_sq * f_idx.unsqueeze(0)).sum(dim=1, keepdim=True) / mel_sum
    f_diff_sq = (f_idx.unsqueeze(0) - centroid)**2
    bandwidth = torch.sqrt((mel_sq * f_diff_sq).sum(dim=1, keepdim=True) / mel_sum)
    rms = torch.sqrt((mel_sq**2).mean(dim=1, keepdim=True))
    augmented = torch.cat([mel_sq, delta_44, flux, centroid, bandwidth, rms], dim=1)
    return augmented.unsqueeze(0)

def temporal_nms(candidates, window_ms=100.0):
    if not candidates:
        return []
    sorted_cands = sorted(candidates, key=lambda x: x["prob"], reverse=True)
    keep = []
    while sorted_cands:
        best = sorted_cands.pop(0)
        keep.append(best)
        sorted_cands = [c for c in sorted_cands if not (c["type"] == best["type"] and abs(c["time"] - best["time"]) < window_ms)]
    return sorted(keep, key=lambda x: x["time"])

def generate_batch(track_file, window_ms=100.0, out_root=None):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[+] Device: {device}")
    
    model = TransformerCausalDecoder(d_model=256, num_layers=4, d_audio=128, d_target=8)
    checkpoint_name = os.environ.get("RL_CKPT", "transformer_phase12b_rl_ep5.pt")
    ckpt_path = f"/Volumes/Second-Brain-1/AI/Synth/models/checkpoints/{checkpoint_name}"
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    if out_root is None:
        out_root = Path("/Volumes/Second-Brain-1/AI/Synth/evaluation/phase12b/gold_standard")
    else:
        out_root = Path(out_root)
    os.makedirs(out_root, exist_ok=True)
    
    with open(track_file, "r") as f:
        paths = [line.strip() for line in f if line.strip()]

    print(f"[+] Generating {len(paths)} tracks with {window_ms}ms NMS...")
    
    for feat_path in tqdm(paths):
        uuid = Path(feat_path).stem
        data = np.load(feat_path)
        raw_mel = torch.from_numpy(data["audio_mel"]).float().unsqueeze(0).to(device)
        audio = compute_augmented_features(raw_mel)
        T_len = audio.shape[1]
        
        targets = torch.zeros((1, T_len, 8)).to(device)
        results = {}
        
        for diff_idx in [0, 1, 2, 3, 4]:
            diff_str = ["Easy", "Normal", "Hard", "Expert", "Master"][diff_idx]
            diff_tensor = torch.tensor([diff_idx]).to(device)
            
            with torch.no_grad():
                preds = model(audio, targets, diff_tensor)
                
            pres_prob = torch.sigmoid(preds["presence_logits"][0]).cpu().numpy()
            pos_pred = preds["position_pred"][0].cpu().numpy()
            
            candidates = []
            for t_idx in range(T_len):
                for h in range(2):
                    if pres_prob[t_idx, h] > 0.5:
                        candidates.append({
                            "time": t_idx * 10.0,
                            "type": int(h),
                            "prob": float(pres_prob[t_idx, h]),
                            "pos": pos_pred[t_idx, h*2:(h+1)*2]
                        })
            
            cleaned = temporal_nms(candidates, window_ms=NMS_WINDOW)
            
            notes = []
            for n in cleaned:
                notes.append({
                    "time": float(n["time"]),
                    "type": int(n["type"]),
                    "x": float(n["pos"][0] * 3.0),
                    "y": float(n["pos"][1] * 2.0)
                })
            results[diff_str] = notes

        # Save as a single .json bundle for the UUID
        out_path = out_root / f"{uuid}.json"
        with open(out_path, "w") as f:
            json.dump({
                "uuid": uuid,
                "version": VERSION_TAG,
                "difficulties": results
            }, f)

if __name__ == "__main__":
    # Check for CLI args to run full batch or smoke test
    import argparse
    parser = argparse.ArgumentParser(description="SynthRiders Production Generation Engine (v12b)")
    parser.add_argument("--batch", type=str, help="Path to track list file")
    parser.add_argument("--auto-scan", action="store_true", help="Scan the audio_features directory for all unmapped tracks")
    parser.add_argument("--outdir", type=str, default="/Volumes/Second-Brain-1/AI/Synth/evaluation/phase12b/gold_standard", help="Output directory")
    args = parser.parse_args()

    if args.auto_scan:
        audio_dir = Path("/Volumes/Second-Brain-1/AI/Synth/dataset/audio_features")
        all_tracks = list(audio_dir.glob("*.npz"))
        # Save to temp list for the generator
        temp_list = "/tmp/autoscan_tracks.txt"
        with open(temp_list, "w") as f:
            for t in all_tracks:
                f.write(str(t) + "\n")
        generate_batch(temp_list, window_ms=NMS_WINDOW, out_root=args.outdir)
    elif args.batch:
        generate_batch(args.batch, window_ms=NMS_WINDOW, out_root=args.outdir)
    else:
        # Default fallback to Gold Standard subset
        track_list_file = "/Volumes/Second-Brain-1/AI/Synth/evaluation/gold_standard_tracks.txt"
        generate_batch(track_list_file, window_ms=NMS_WINDOW, out_root=args.outdir)
