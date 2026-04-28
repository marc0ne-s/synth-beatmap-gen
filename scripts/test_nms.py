#!/usr/bin/env python3
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path("/Volumes/Second-Brain-1/AI/Synth")))

from src.models.transformer import TransformerCausalDecoder

def compute_augmented_features(mel: torch.Tensor) -> torch.Tensor:
    """Replicating the StreamingDataset augmentation logic for 80 -> 128 conversion."""
    # mel: (1, T, 80)
    T, num_freq = mel.shape[1], mel.shape[2]
    mel_sq = mel[0] # (T, 80)
    
    # 1. Delta
    shifted_mel = torch.cat([torch.zeros(1, num_freq, device=mel.device), mel_sq[:-1, :]], dim=0)
    delta = mel_sq - shifted_mel
    
    # Delta 44
    # Match dataset: (1, T, 80) -> pool over 80 -> (1, T, 44)
    delta_cpu = delta.cpu().unsqueeze(0) # (1, T, 80)
    delta_44 = F.adaptive_avg_pool1d(delta_cpu, 44).squeeze(0).to(mel.device) # (T, 44)
    
    # 2. Flux
    flux = F.relu(delta).sum(dim=1, keepdim=True) / 10.0
    
    # 3. Centroid
    f_idx = torch.arange(num_freq, dtype=torch.float32, device=mel.device) / (num_freq - 1)
    mel_sum = mel_sq.sum(dim=1, keepdim=True) + 1e-8
    centroid = (mel_sq * f_idx.unsqueeze(0)).sum(dim=1, keepdim=True) / mel_sum
    
    # 4. Bandwidth
    f_diff_sq = (f_idx.unsqueeze(0) - centroid)**2
    bandwidth = torch.sqrt((mel_sq * f_diff_sq).sum(dim=1, keepdim=True) / mel_sum)
    
    # 5. RMS
    rms = torch.sqrt((mel_sq**2).mean(dim=1, keepdim=True))
    
    # 128-D: 80 (mel) + 44 (delta_44) + 1 (flux) + 1 (centroid) + 1 (bandwidth) + 1 (rms) = 128
    augmented = torch.cat([mel_sq, delta_44, flux, centroid, bandwidth, rms], dim=1)
    return augmented.unsqueeze(0)

def temporal_nms(notes, window_ms=80.0):
    if not notes:
        return []
    notes = sorted(notes, key=lambda x: x["prob"], reverse=True)
    keep = []
    while notes:
        best = notes.pop(0)
        keep.append(best)
        notes = [n for n in notes if not (n["type"] == best["type"] and abs(n["time"] - best["time"]) < window_ms)]
    return sorted(keep, key=lambda x: x["time"])

def check_nms():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[+] Device: {device}")
    
    model = TransformerCausalDecoder(d_model=256, num_layers=4, d_audio=128, d_target=8)
    ckpt_path = "/Volumes/Second-Brain-1/AI/Synth/models/checkpoints/transformer_phase12b_ep5.pt"
    
    print(f"[+] Loading {ckpt_path}...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Da da dance
    feat_path = "/Volumes/Second-Brain-1/AI/Synth/dataset/audio_features/9be318efb916770f.npz"
    data = np.load(feat_path)
    raw_mel = torch.from_numpy(data["audio_mel"]).float().unsqueeze(0).to(device)
    
    print(f"[+] Augmenting features (80 -> 128)...")
    audio = compute_augmented_features(raw_mel)
    
    T = audio.shape[1]
    targets = torch.zeros((1, T, 8)).to(device)
    diff = torch.tensor([4]).to(device)
    
    print(f"[+] Generating raw predictions for Da da dance...")
    with torch.no_grad():
        preds = model(audio, targets, diff)
        
    pres_logits = preds["presence_logits"][0].cpu().numpy()
    pos_pred = preds["position_pred"][0].cpu().numpy()
    
    pres_prob = 1.0 / (1.0 + np.exp(-pres_logits))
    threshold = 0.5
    
    candidates = []
    for t_idx in range(T):
        for h in range(2):
            if pres_prob[t_idx, h] > threshold:
                candidates.append({
                    "time": t_idx * 10.0,
                    "type": h,
                    "prob": float(pres_prob[t_idx, h]),
                    "pos": pos_pred[t_idx, h*2:(h+1)*2].tolist()
                })
    
    print(f"[!] Raw candidates above {threshold}: {len(candidates)}")
    
    # NMS Tests
    for win in [40.0, 80.0, 150.0, 200.0]:
        cleaned = temporal_nms(candidates, window_ms=win)
        print(f"[*] NMS ({win:3.0f}ms) -> {len(cleaned):4d} notes (Total)")

if __name__ == "__main__":
    check_nms()
