#!/usr/bin/env python3
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path("/Volumes/Second-Brain-1/AI/Synth")))

from src.models.hybrid_generator import HybridCascadeModel
from src.data.streaming_loader import get_streaming_loader

def evaluate_hybrid():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[+] Device: {device}")
    
    model = HybridCascadeModel().to(device)
    base_ckpt = "/Volumes/Second-Brain-1/AI/Synth/models/checkpoints/best_model.pt"
    # Wait for ep5
    trans_ckpt = "/Volumes/Second-Brain-1/AI/Synth/models/checkpoints/hybrid_cascade_ep5.pt"
    
    if not os.path.exists(trans_ckpt):
        # Fallback to whatever is available
        import glob
        ckpts = glob.glob("/Volumes/Second-Brain-1/AI/Synth/models/checkpoints/hybrid_cascade_ep*.pt")
        if not ckpts:
            print(f"[!] No hybrid cascade checkpoint found!")
            return
        trans_ckpt = max(ckpts)
        
    print(f"[+] Loading {base_ckpt} and {trans_ckpt}...")
    model.load_states(
        baseline_ckpt=base_ckpt,
        transformer_ckpt=trans_ckpt,
        device=device
    )
    model.eval()

    features_dir = "/Volumes/Second-Brain-1/AI/Synth/dataset/features"
    audio_dir = "/Volumes/Second-Brain-1/AI/Synth/dataset/audio_features"
    _, val_loader = get_streaming_loader(
        features_dir=features_dir,
        audio_dir=audio_dir,
        difficulties=["Hard", "Expert", "Master"],
        batch_size=8,
        num_maps=2500
    )

    total_gt = 0
    total_pred = 0
    total_matches = 0
    total_pos_mse = 0.0
    total_valid_pos = 0

    frame_ms = 10.0
    match_tolerance_frames = int(50.0 / frame_ms)

    print("[+] Evaluating Cascaded Pipeline on Validation Stream...")
    
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
            pos_pred = preds["position_pred"]
            
            p_probs = torch.sigmoid(pres_logits)
            p_binary = (p_probs > 0.5).float()
            t_binary = targets[..., 0:2]
            
            B, T, _ = t_binary.shape
            mask = torch.arange(T, device=device).unsqueeze(0).expand(B, T) < lengths.unsqueeze(1)
            
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
    
    print("=" * 40)
    print("PHASE 11 QUANTITATIVE METRICS (CASCADE)")
    print(f"Recall@50ms:    {recall:.2f}%")
    print(f"Precision@50ms: {precision:.2f}%")
    print(f"Position MSE:   {pos_mse:.4f}")
    print("=" * 40)
    
    results = {
        "Recall@50ms": recall,
        "Precision@50ms": precision,
        "Position_MSE": pos_mse,
        "Total_GT_Notes": total_gt,
        "Total_Predicted": total_pred,
        "Total_Matches": total_matches
    }
    
    out_dir = "/Volumes/Second-Brain-1/AI/Synth/evaluation/phase11"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "quantitative_results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"[+] Written to {out_dir}/quantitative_results.json")

if __name__ == "__main__":
    evaluate_hybrid()
