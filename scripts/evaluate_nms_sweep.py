#!/usr/bin/env python3
import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path("/Volumes/Second-Brain-1/AI/Synth")))

from src.models.transformer import TransformerCausalDecoder
from src.data.streaming_loader import get_streaming_loader

def compute_augmented_features(mel: torch.Tensor) -> torch.Tensor:
    """Replicating the StreamingDataset augmentation logic for 80 -> 128 conversion."""
    T, num_freq = mel.shape[1], mel.shape[2]
    mel_sq = mel[0]
    shifted_mel = torch.cat([torch.zeros(1, num_freq, device=mel.device), mel_sq[:-1, :]], dim=0)
    delta = mel_sq - shifted_mel
    # Use CPU for adaptive pool to avoid MPS divisional restriction
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

def temporal_nms(candidates, window_ms=80.0):
    if not candidates:
        return []
    # Sort by confidence descending
    sorted_cands = sorted(candidates, key=lambda x: x["prob"], reverse=True)
    keep = []
    while sorted_cands:
        best = sorted_cands.pop(0)
        keep.append(best)
        # Suppress nearby same-type
        sorted_cands = [c for c in sorted_cands if not (c["type"] == best["type"] and abs(c["time"] - best["time"]) < window_ms)]
    return sorted(keep, key=lambda x: x["time"])

def evaluate_nms_sweep():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = TransformerCausalDecoder(d_model=256, num_layers=4, d_audio=128, d_target=8)
    ckpt_path = "/Volumes/Second-Brain-1/AI/Synth/models/checkpoints/transformer_phase12b_ep5.pt"
    
    print(f"[+] Loading {ckpt_path}...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Full validation set
    features_dir = "/Volumes/Second-Brain-1/AI/Synth/dataset/features"
    audio_dir = "/Volumes/Second-Brain-1/AI/Synth/dataset/audio_features"
    _, val_loader = get_streaming_loader(
        features_dir=features_dir,
        audio_dir=audio_dir,
        difficulties=["Hard", "Expert", "Master"],
        batch_size=4, # Smaller batch for NMS overhead
        num_maps=8861
    )

    windows = [60, 80, 100, 120, 150]
    total_gt = 0
    # Metrics per window
    win_matches_recall = {w: 0 for w in windows}
    win_matches_precision = {w: 0 for w in windows}
    win_preds = {w: 0 for w in windows}

    match_tolerance = 50.0 # ms

    print(f"[+] Sweeping NMS Windows: {windows}")
    with torch.no_grad():
        for batch_idx, (audio, targets, diff, lengths) in enumerate(tqdm(val_loader)):
            audio = audio.to(device)
            targets = targets.to(device)
            diff = diff.to(device)
            lengths = lengths.to(device)
            
            # Predict
            preds = model(audio, targets, diff)
            pres_logits = preds["presence_logits"]
            pres_prob = torch.sigmoid(pres_logits).cpu().numpy()
            
            t_binary = targets[..., 0:2].cpu().numpy()
            B = audio.shape[0]
            
            for b in range(B):
                length = lengths[b].item()
                # 1. Extract GT
                gt_notes = []
                for h in range(2):
                    gt_frames = np.where(t_binary[b, :length, h] == 1)[0]
                    for f in gt_frames:
                        gt_notes.append({"time": f * 10.0, "type": h})
                total_gt += len(gt_notes)
                
                # 2. Extract Raw Candidates (t=0.5)
                candidates = []
                for t_idx in range(length):
                    for h in range(2):
                        if pres_prob[b, t_idx, h] > 0.5:
                            candidates.append({
                                "time": t_idx * 10.0,
                                "type": h,
                                "prob": float(pres_prob[b, t_idx, h])
                            })
                
                # 3. Apply NMS for each window
                for w in windows:
                    cleaned = temporal_nms(candidates, window_ms=float(w))
                    win_preds[w] += len(cleaned)
                    
                    # Compute Match@50ms for this window
                    # 3a. Recall (GT-centric)
                    matches_for_recall = 0
                    for gt in gt_notes:
                        for p in cleaned:
                            if p["type"] != gt["type"]: continue
                            if abs(p["time"] - gt["time"]) <= match_tolerance:
                                matches_for_recall += 1
                                break
                    win_matches_recall[w] += matches_for_recall

                    # 3b. Precision (Pred-centric)
                    matches_for_precision = 0
                    for p in cleaned:
                        for gt in gt_notes:
                            if p["type"] != gt["type"]: continue
                            if abs(p["time"] - gt["time"]) <= match_tolerance:
                                matches_for_precision += 1
                                break
                    win_matches_precision[w] += matches_for_precision
            
            if batch_idx >= 50:
                break

    print("\n" + "="*50)
    print("NMS WINDOW SWEEP RESULTS")
    print("="*50)
    for w in windows:
        recall = (win_matches_recall[w] / total_gt * 100) if total_gt > 0 else 0
        precision = (win_matches_precision[w] / win_preds[w] * 100) if win_preds[w] > 0 else 0
        f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0
        print(f"Window {w:3d}ms | Recall: {recall:5.1f}% | Precision: {precision:5.1f}% | F1: {f1:5.1f}%")
    print("="*50)

if __name__ == "__main__":
    evaluate_nms_sweep()
