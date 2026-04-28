#!/usr/bin/env python3
"""
Quick eval for Phase 12 pure transformer checkpoints.
Usage: python scripts/evaluate_transformer_full.py --epoch 4
"""
import os, sys, json, argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path("/Volumes/Second-Brain-1/AI/Synth")))

from src.models.transformer import TransformerCausalDecoder
from src.data.streaming_loader import get_streaming_loader

def evaluate(checkpoint_path: str):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[+] Device: {device}")
    print(f"[+] Loading {checkpoint_path}...")
    
    model = TransformerCausalDecoder(
        d_model=256,
        num_layers=4,
        d_audio=128,
        d_target=8,
    ).to(device)
    
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print("[+] Model loaded.")
    
    features_dir = "/Volumes/Second-Brain-1/AI/Synth/dataset/features"
    audio_dir = "/Volumes/Second-Brain-1/AI/Synth/dataset/audio_features"
    _, val_loader = get_streaming_loader(
        features_dir=features_dir,
        audio_dir=audio_dir,
        difficulties=["Hard", "Expert", "Master"],
        batch_size=8,
        num_maps=2500,
    )
    
    total_gt = 0
    total_pred = 0
    total_matches = 0
    total_pos_mse = 0.0
    total_valid_pos = 0
    
    frame_ms = 10.0
    match_tolerance_frames = int(50.0 / frame_ms)
    
    print("[+] Evaluating on validation stream...")
    
    with torch.no_grad():
        for batch_idx, (audio, targets, diff, lengths) in enumerate(tqdm(val_loader, total=63)):
            audio = audio.to(device)
            targets = targets.to(device)
            diff = diff.long().to(device)
            context = torch.amp.autocast(device_type="mps", dtype=torch.bfloat16) if device.type == "mps" else __import__('contextlib').nullcontext()
            with context:
                preds = model(audio, targets, diff.squeeze(-1) if diff.dim() > 1 else diff)
            
            # preds shape depends on model — inspect first batch
            if batch_idx == 0:
                print(f"[i] prediction type: {type(preds)}")
                if isinstance(preds, dict):
                    for k, v in preds.items():
                        print(f"    {k}: {v.shape if hasattr(v, 'shape') else type(v)}")
                elif hasattr(preds, 'shape'):
                    print(f"    shape: {preds.shape}")
            
            # Handle different output formats
            if isinstance(preds, dict):
                pres_logits = preds.get("presence_logits")
                pos_pred = preds.get("position_pred")
            elif isinstance(preds, torch.Tensor):
                # Split: first 2 dims presence, remainder position
                pres_logits = preds[..., :2]
                pos_pred = preds[..., 2:]
            else:
                print(f"[!] Unexpected pred type: {type(preds)}")
                break
            
            if pres_logits is None:
                print("[!] Could not find presence logits")
                break
            
            p_probs = torch.sigmoid(pres_logits)
            p_binary = (p_probs > 0.5).float()
            t_binary = targets[..., 0:2]
            
            B, T, _ = t_binary.shape
            
            # Cast position predictions to float32 for CPU operations
            if pos_pred is not None:
                pos_pred = pos_pred.float()
            for b in range(B):
                pred_frames = p_binary[b, :, 0].cpu().numpy()
                true_frames = t_binary[b, :, 0].cpu().numpy()
                
                pred_times = np.where(pred_frames > 0.5)[0]
                true_times = np.where(true_frames > 0.5)[0]
                
                total_gt += len(true_times)
                total_pred += len(pred_times)
                
                matches = 0
                for pt in pred_times:
                    matched = np.where(np.abs(true_times - pt) <= match_tolerance_frames)[0]
                    if len(matched) > 0:
                        matches += 1
                        true_times = np.delete(true_times, matched[0])
                
                total_matches += matches
                
                # Position MSE on predicted notes
                if pos_pred is not None:
                    for pt in pred_times:
                        if pt < pos_pred.shape[1]:
                            pred_pos = pos_pred[b, pt, :3].cpu().numpy()  # take first 3 dims for x,y,z
                            
                            # Find closest ground truth note in time tolerance
                            true_indices = np.where(np.abs(np.where(true_frames > 0.5)[0] - pt) <= match_tolerance_frames)[0]
                            if len(true_indices) > 0:
                                true_t = np.where(true_frames > 0.5)[0][true_indices[0]]
                                true_pos = targets[b, true_t, 2:5].cpu().numpy() if targets.shape[-1] >= 5 else np.zeros(3)
                                total_pos_mse += np.mean((pred_pos - true_pos) ** 2)
                                total_valid_pos += 1
    
    recall = (total_matches / total_gt * 100) if total_gt > 0 else 0
    precision = (total_matches / total_pred * 100) if total_pred > 0 else 0
    pos_mse = (total_pos_mse / total_valid_pos) if total_valid_pos > 0 else 0
    
    print("=" * 50)
    print(f"Recall@50ms:     {recall:.2f}%")
    print(f"Precision@50ms:  {precision:.2f}%")
    print(f"Position MSE:    {pos_mse:.4f}")
    print(f"Total GT Notes:  {total_gt}")
    print(f"Total Predicted: {total_pred}")
    print(f"Total Matches:   {total_matches}")
    print("=" * 50)
    
    results = {
        "checkpoint": checkpoint_path,
        "Recall@50ms": float(recall),
        "Precision@50ms": float(precision),
        "Position_MSE": float(pos_mse),
        "Total_GT_Notes": int(total_gt),
        "Total_Predicted": int(total_pred),
        "Total_Matches": int(total_matches),
    }
    
    out_dir = Path("/Volumes/Second-Brain-1/AI/Synth/evaluation/phase12")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "transformer_full_ep4_eval.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[+] Results written to {out_file}")
    
    return recall, precision, pos_mse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=4, help="Epoch number to evaluate")
    args = parser.parse_args()
    
    ckpt = f"/Volumes/Second-Brain-1/AI/Synth/models/checkpoints/transformer_full_ep{args.epoch}.pt"
    evaluate(ckpt)