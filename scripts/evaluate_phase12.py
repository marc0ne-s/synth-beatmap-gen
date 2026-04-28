#!/usr/bin/env python3
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path("/Volumes/Second-Brain-1/AI/Synth")))

from src.models.transformer import TransformerCausalDecoder
from src.data.streaming_loader import get_streaming_loader

def evaluate_phase12(threshold=0.5, ckpt_path=None):
    print("=====================================================")
    print(f"  PHASE 12 EVALUATION (Threshold: {threshold})")
    print("=====================================================")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[+] Device: {device}")
    
    model = TransformerCausalDecoder(d_model=256, num_layers=4, d_audio=128, d_target=8)
    if ckpt_path is None:
        ckpt_path = "/Volumes/Second-Brain-1/AI/Synth/models/checkpoints/transformer_full_ep9.pt"
    
    if not os.path.exists(ckpt_path):
        print(f"[!] Checkpoint not found: {ckpt_path}")
        return
        
    print(f"[+] Loading {ckpt_path}...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Load 8861 corpus val stream
    features_dir = "/Volumes/Second-Brain-1/AI/Synth/dataset/features"
    audio_dir = "/Volumes/Second-Brain-1/AI/Synth/dataset/audio_features"
    _, val_loader = get_streaming_loader(
        features_dir=features_dir,
        audio_dir=audio_dir,
        difficulties=["Hard", "Expert", "Master"],
        batch_size=8,
        num_maps=8861
    )

    total_gt = 0
    total_pred = 0
    total_matches = 0
    total_pos_mse = 0.0
    total_valid_pos = 0

    frame_ms = 10.0
    match_tolerance_frames = int(50.0 / frame_ms)

    print("[+] Iterating over FULL Validation Split...")
    
    with torch.no_grad():
        for batch_idx, (audio, targets, diff, lengths) in enumerate(tqdm(val_loader)):
            audio = audio.to(device)
            targets = targets.to(device)
            diff = diff.to(device)
            lengths = lengths.to(device)
            
            context = torch.amp.autocast(device_type="mps", dtype=torch.bfloat16) if device.type == "mps" else __import__('contextlib').nullcontext()
            with context:
                preds = model(audio, targets, diff)
                
            pres_logits = preds["presence_logits"]
            pos_pred = preds["position_pred"] # Model now has Tanh natively
            
            p_probs = torch.sigmoid(pres_logits)
            p_binary = (p_probs > threshold).float()
            t_binary = targets[..., 0:2]
            
            B, T, _ = t_binary.shape
            
            for b in range(B):
                length = lengths[b].item()
                for h in range(2): 
                    gt_frames = torch.where(t_binary[b, :length, h] == 1)[0].cpu().numpy()
                    pred_frames = torch.where(p_binary[b, :length, h] == 1)[0].cpu().numpy()
                    
                    total_gt += len(gt_frames)
                    total_pred += len(pred_frames)
                    
                    matched_preds = set()
                    for gt_f in gt_frames:
                        best_dist = match_tolerance_frames + 1
                        best_p = -1
                        for p_f in pred_frames:
                            if p_f in matched_preds:
                                continue
                            dist = abs(gt_f - p_f)
                            if dist <= match_tolerance_frames and dist < best_dist:
                                best_dist = dist
                                best_p = p_f
                        
                        if best_p != -1:
                            matched_preds.add(best_p)
                            total_matches += 1
                            
                            # Positions are in indices 2:6 (H0_x, H0_y, H1_x, H1_y)
                            if h == 0:
                                t_pos = targets[b, gt_f, 2:4]
                                p_pos = pos_pred[b, best_p, 0:2]
                            else:
                                t_pos = targets[b, gt_f, 4:6]
                                p_pos = pos_pred[b, best_p, 2:4]
                            
                            se = torch.sum((p_pos - t_pos)**2).item()
                            total_pos_mse += se
                            total_valid_pos += 1

    recall = (total_matches / total_gt * 100) if total_gt > 0 else 0
    precision = (total_matches / total_pred * 100) if total_pred > 0 else 0
    pos_mse = (total_pos_mse / total_valid_pos) if total_valid_pos > 0 else 0
    
    print("=" * 45)
    print("PHASE 12 RESULTS (CONVERGED)")
    print(f"Recall@50ms:    {recall:.2f}%")
    print(f"Precision@50ms: {precision:.2f}%")
    print(f"Position MSE:   {pos_mse:.4f}")
    print("=" * 45)
    
    results = {
        "Threshold": float(threshold),
        "Recall@50ms": float(recall),
        "Precision@50ms": float(precision),
        "Position_MSE": float(pos_mse),
        "Total_GT_Notes": int(total_gt),
        "Total_Predicted": int(total_pred),
        "Total_Matches": int(total_matches),
        "Checkpoint": str(ckpt_path)
    }
    
    out_dir = "/Volumes/Second-Brain-1/AI/Synth/evaluation/phase12"
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_t{int(threshold*100)}"
    # Add checkpoint name to suffix to avoid overwrites during sweep
    ckpt_name = Path(ckpt_path).stem
    filename = f"audit_{ckpt_name}_t{int(threshold*100)}.json"
    
    with open(os.path.join(out_dir, filename), "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"[+] Audit Saved to {out_dir}/{filename}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    evaluate_phase12(threshold=args.threshold, ckpt_path=args.checkpoint)
