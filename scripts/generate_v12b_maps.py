#!/usr/bin/env python3
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path

sys.path.insert(0, str(Path("/Volumes/Second-Brain-1/AI/Synth")))

from src.models.transformer import TransformerCausalDecoder

def compute_augmented_features(mel: torch.Tensor) -> torch.Tensor:
    """Standard 80 -> 128 augmentation."""
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

def temporal_nms(candidates, window_ms=60.0):
    if not candidates:
        return []
    sorted_cands = sorted(candidates, key=lambda x: x["prob"], reverse=True)
    keep = []
    while sorted_cands:
        best = sorted_cands.pop(0)
        keep.append(best)
        sorted_cands = [c for c in sorted_cands if not (c["type"] == best["type"] and abs(c["time"] - best["time"]) < window_ms)]
    return sorted(keep, key=lambda x: x["time"])

def write_synth(out_path, synth_dict):
    with open(out_path, "w") as f:
        json.dump(synth_dict, f, indent=4)

def generate_v12b_maps(tracks, window_ms=60.0):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[+] Device: {device}")
    
    model = TransformerCausalDecoder(d_model=256, num_layers=4, d_audio=128, d_target=8)
    ckpt_path = "/Volumes/Second-Brain-1/AI/Synth/models/checkpoints/transformer_phase12b_ep5.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    out_root = Path("/Volumes/Second-Brain-1/AI/Synth/evaluation/phase12b/generated_maps")
    
    for t in tracks:
        print(f"\n[+] Processing: {t['name']}")
        feat_path = f"/Volumes/Second-Brain-1/AI/Synth/dataset/audio_features/{t['uuid']}.npz"
        if not os.path.exists(feat_path):
            print(f"  [!] Missing audio features: {feat_path}")
            continue
            
        data = np.load(feat_path)
        raw_mel = torch.from_numpy(data["audio_mel"]).float().unsqueeze(0).to(device)
        audio = compute_augmented_features(raw_mel)
        T_len = audio.shape[1]
        
        # Batch inference 
        targets = torch.zeros((1, T_len, 8)).to(device)
        # Gen for all 5 difficulties
        for diff_idx in [0, 1, 2, 3, 4]:
            diff_str = ["Easy", "Normal", "Hard", "Expert", "Master"][diff_idx]
            diff_tensor = torch.tensor([diff_idx]).to(device)
            
            with torch.no_grad():
                preds = model(audio, targets, diff_tensor)
                
            pres_prob = torch.sigmoid(preds["presence_logits"][0]).cpu().numpy()
            pos_pred = preds["position_pred"][0].cpu().numpy()
            
            # Extract candidates
            threshold = 0.5
            candidates = []
            for t_idx in range(T_len):
                for h in range(2):
                    if pres_prob[t_idx, h] > threshold:
                        candidates.append({
                            "time": t_idx * 10.0,
                            "type": h,
                            "prob": float(pres_prob[t_idx, h]),
                            "pos": pos_pred[t_idx, h*2:(h+1)*2]
                        })
            
            # Application of the 60ms "Resurrection" Window
            cleaned = temporal_nms(candidates, window_ms=window_ms)
            
            # Build .synth structure
            notes_list = []
            for n in cleaned:
                notes_list.append({
                    "time": n["time"],
                    "type": n["type"],
                    "x": float(n["pos"][0] * 3.0),
                    "y": float(n["pos"][1] * 2.0)
                })
            
            synth_dict = {
                "name": f"[AIv12b.150ms] {t['name']}",
                "difficulty": diff_str,
                "notes": notes_list
            }
            
            out_dir = out_root / t["name"].replace(" ", "_")
            os.makedirs(out_dir, exist_ok=True)
            out_path = out_dir / f"{diff_str}.synth"
            write_synth(out_path, synth_dict)
            print(f"  -> {diff_str}: {len(cleaned)} notes generated.")

if __name__ == "__main__":
    # Stage 2: 10-Map Diversity Sweep (80-210 BPM)
    diversity_tracks = [
        {"name": "BPM_97", "uuid": "9f2d7c19ab35f5ae"},
        {"name": "BPM_110", "uuid": "e1000fef2c223f10"},
        {"name": "BPM_120", "uuid": "1492bcfbca354213"},
        {"name": "BPM_130", "uuid": "b1755a0bc939d3d7"},
        {"name": "BPM_142", "uuid": "1350397c2d97a7c1"},
        {"name": "BPM_150", "uuid": "ffa176618e7d2d30"},
        {"name": "BPM_160", "uuid": "6877eab99b06a020"},
        {"name": "BPM_175", "uuid": "7247c27d84ce6be2"},
        {"name": "BPM_190", "uuid": "016ff3d7b5711bfd"},
        {"name": "BPM_210", "uuid": "1944c628b0e201b3"}
    ]
    generate_v12b_maps(diversity_tracks, window_ms=150.0)
