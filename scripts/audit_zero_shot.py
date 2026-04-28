import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.transformer import TransformerCausalDecoder
from src.data.streaming_loader import StreamingSynthDataset, streaming_collate_fn

def get_disjoint_audit_loader(features_dir, audio_dir, num_maps=100):
    """
    Carefully isolates 100 'Master' maps that were definitively NOT included
    in the Phase 5 (2500 map) training shuffle.
    """
    feat_d = Path(features_dir)
    aud_d = Path(audio_dir)
    
    all_files = list(feat_d.glob("*.npz"))
    
    # 1. Reconstruct the exact 2500-map Phase 5 pool
    valid_files_phase5 = []
    for f in all_files:
        diff_str = f.stem.rsplit("_", 1)[-1] if "_" in f.stem else "Hard"
        if diff_str in ["Hard", "Expert", "Master"]:
            valid_files_phase5.append(f)
            
    np.random.seed(42)  # MUST match streaming_loader.py exactly
    np.random.shuffle(valid_files_phase5)
    phase5_pool = set(valid_files_phase5[:2500])
    
    # 2. Extract out of pool Master maps
    disjoint_master_files = []
    for f in all_files:
        diff_str = f.stem.rsplit("_", 1)[-1] if "_" in f.stem else "Hard"
        if diff_str == "Master" and f not in phase5_pool:
            disjoint_master_files.append(f)
            
    print(f"[GhostAudit] Found {len(disjoint_master_files)} disjoint Master maps.")
    
    # 3. Formulate the explicit 100-map Stress subset
    np.random.seed(999) # New seed for audit stochasticity
    np.random.shuffle(disjoint_master_files)
    audit_files = disjoint_master_files[:num_maps]
    
    print(f"[GhostAudit] Compiled exact {len(audit_files)}-map Ghost sequence.")
    
    dataset = StreamingSynthDataset(audit_files, aud_d)
    return DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=streaming_collate_fn, num_workers=0)


def perform_ghost_audit():
    print("=====================================================")
    print("  PHASE 6 AUDIT: ZERO-SHOT MAE INFERENCE")
    print("=====================================================")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[X] Targeting Accelerator: {device}")
    
    loader = get_disjoint_audit_loader(
        features_dir="/Volumes/Second-Brain-1/AI/Synth/dataset/features",
        audio_dir="/Volumes/Second-Brain-1/AI/Synth/dataset/audio_features",
        num_maps=100
    )
    
    model = TransformerCausalDecoder(d_model=256, num_layers=4, d_audio=128, d_target=8)
    model.load_state_dict(torch.load("models/checkpoints/transformer_pilot_ep5.pt", weights_only=True))
    model.to(device)
    model.eval()
    
    # Metrics Accumulators
    total_active_frames = 0
    total_pos_mae = 0.0
    total_vel_mae = 0.0
    
    total_true_pos = 0
    total_false_pos = 0
    total_false_neg = 0

    print("[X] Initiating Structural Geometry Scan...")
    
    context = torch.amp.autocast(device_type="mps", dtype=torch.bfloat16) if device.type == "mps" else __import__('contextlib').nullcontext()
    
    with torch.no_grad():
        for batch_idx, (audio, targets, diff, lengths) in enumerate(loader):
            audio = audio.to(device)
            targets = targets.to(device)
            diff = diff.to(device)
            lengths = lengths.to(device)
            
            with context:
                preds = model(audio, targets, diff)
                
            pres_logits = preds["presence_logits"]
            pos_pred = preds["position_pred"]
            vel_pred = preds["velocity_pred"]
            
            target_pres = targets[..., 0:2]
            target_pos = targets[..., 2:6]
            target_vel = targets[..., 6:8]
            
            B, T, _ = target_pres.shape
            mask = torch.arange(T, device=device).unsqueeze(0).expand(B, T) < lengths.unsqueeze(1)
            
            # 1. Presence Evaluation (Recall/Precision)
            pres_probs = torch.sigmoid(pres_logits)
            pres_pred_binary = (pres_probs > 0.5) & mask.unsqueeze(-1)
            active_target_binary = (target_pres > 0.5) & mask.unsqueeze(-1)
            
            total_true_pos += (pres_pred_binary & active_target_binary).sum().item()
            total_false_pos += (pres_pred_binary & ~active_target_binary).sum().item()
            total_false_neg += (~pres_pred_binary & active_target_binary).sum().item()
            
            # 2. Geometric MAE Evaluation
            active_mask = active_target_binary # Only evaluate where genuine music requires geometry
            active_mask_pos = torch.cat([active_mask[..., 0:1], active_mask[..., 0:1], 
                                         active_mask[..., 1:2], active_mask[..., 1:2]], dim=-1)
            
            frames_counted = active_mask_pos.sum().item()
            if frames_counted > 0:
                abs_pos_err = torch.abs(pos_pred - target_pos) * active_mask_pos
                total_pos_mae += abs_pos_err.sum().item()
                
                abs_vel_err = torch.abs(vel_pred - target_vel) * active_mask
                total_vel_mae += abs_vel_err.sum().item()
                
                total_active_frames += frames_counted / 2.0  # Normalized to hands
                
    # Final Metric Synthesis
    precision = total_true_pos / (total_true_pos + total_false_pos + 1e-8)
    recall = total_true_pos / (total_true_pos + total_false_neg + 1e-8)
    # The positions are scaled tanh(-1, 1). To get real world proxy grid space MAE we mult by 3.
    avg_pos_mae = (total_pos_mae / (total_active_frames * 4 + 1e-8)) * 3.0 
    # Velocity was squeezed tanh(-1, 1) upon input scaled by 2.0. To get physical delta we mult by 2.
    avg_vel_mae = (total_vel_mae / (total_active_frames * 2 + 1e-8)) * 2.0 
    
    print("=====================================================")
    print("            GHOST AUDIT METRIC TENSOR                ")
    print("=====================================================")
    print(f" Stress Test Load: 100 'Master' Maps")
    print(f" Total Active Frames Encountered: {int(total_active_frames):,}\n")
    print(f" Presence Precision: {precision*100:.2f}% (How accurate its guesses were)")
    print(f" Presence Recall:    {recall*100:.2f}% (How many real notes it found)\n")
    print(f" Geometric $\Delta$ Error: {avg_pos_mae:.4f} grid units")
    print(f" Kinetic $\Delta$ Error:   {avg_vel_mae:.4f} differential scale")
    print("=====================================================")

if __name__ == "__main__":
    perform_ghost_audit()
