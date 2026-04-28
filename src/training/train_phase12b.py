import os
import sys
import time
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.streaming_loader import get_streaming_loader
from src.models.transformer import TransformerCausalDecoder

def compute_balanced_loss(predictions, target_features, lengths, gamma=1.0, pos_weight_val=12.0):
    """
    Phase 12b: Precision/Recall Balance Loss.
    Softer Focal gamma (1.0) and aggressive pos_weight (12.0) to buy back recall.
    """
    pres_logits = predictions["presence_logits"]
    pos_pred = predictions["position_pred"]
    vel_pred = predictions["velocity_pred"]
    
    target_pres = target_features[..., 0:2]
    target_pos = target_features[..., 2:6]
    target_vel = target_features[..., 6:8]
    
    B, T, _ = target_pres.shape
    device = pres_logits.device
    mask = torch.arange(T, device=device).unsqueeze(0).expand(B, T) < lengths.unsqueeze(1)
    
    # 1. Presence Loss (Focal Loss with balanced weight)
    # Aggressive pos_weight to prioritize Recall over Precision
    pos_weight = torch.tensor([pos_weight_val, pos_weight_val], device=device)
    bce = F.binary_cross_entropy_with_logits(pres_logits, target_pres, pos_weight=pos_weight, reduction='none')
    
    p = torch.sigmoid(pres_logits)
    p_t = p * target_pres + (1 - p) * (1 - target_pres)
    
    # Softer gamma (1.0) as requested
    focal_weight = (1 - p_t) ** gamma
    focal_loss = focal_weight * bce
    
    focal_loss = focal_loss * mask.unsqueeze(-1)
    loss_pres = focal_loss.sum() / (mask.sum() * 2 + 1e-8)
    
    # 2. Position Loss (Huber Loss) - Maintained for spatial victory
    active_mask = mask.unsqueeze(-1) & (target_pres > 0.5)
    se_pos = F.smooth_l1_loss(pos_pred, target_pos, reduction='none')
    active_mask_pos = torch.cat([active_mask[..., 0:1], active_mask[..., 0:1], 
                                 active_mask[..., 1:2], active_mask[..., 1:2]], dim=-1)
    se_pos = se_pos * active_mask_pos
    loss_pos = se_pos.sum() / (active_mask_pos.sum() + 1e-8)
    
    # 3. Velocity / Intensity Loss (Huber Loss)
    se_vel = F.smooth_l1_loss(vel_pred, target_vel, reduction='none')
    se_vel = se_vel * active_mask
    loss_vel = se_vel.sum() / (active_mask.sum() + 1e-8)
    
    # Balanced Scalar mix
    total = loss_pres + (loss_pos * 10.0) + (loss_vel * 25.0)
    
    return total, loss_pres, loss_pos, loss_vel

def run_phase12b():
    print("=====================================================")
    print("  PHASE 12B: PRECISION/RECALL BALANCE (PRB)")
    print("=====================================================")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[X] Targeting Accelerator: {device}")
    
    features_dir = "/Volumes/Second-Brain-1/AI/Synth/dataset/features"
    audio_dir = "/Volumes/Second-Brain-1/AI/Synth/dataset/audio_features"
    
    BATCH_SIZE = 8
    ACCUM_STEPS = 8
    NUM_EPOCHS = 5 # Forking for refinement
    
    train_loader, val_loader = get_streaming_loader(
        features_dir=features_dir,
        audio_dir=audio_dir,
        difficulties=["Hard", "Expert", "Master"],
        batch_size=BATCH_SIZE,
        num_maps=8861
    )
    
    model = TransformerCausalDecoder(
        d_model=256,
        num_layers=4,
        d_audio=128,
        d_target=8
    ).to(device)
    
    # Fork from Phase 12 Ep 9 (The Spatial Vector)
    ckpt_path = "models/checkpoints/transformer_full_ep9.pt"
    print(f"[X] Forking from {ckpt_path}...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

    # Balanced Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # FIXED OneCycleLR Step Calculation
    # We call scheduler.step() at the end of every accumulation cycle OR the end of the loader.
    steps_per_epoch = (len(train_loader) + ACCUM_STEPS - 1) // ACCUM_STEPS
    total_steps = steps_per_epoch * NUM_EPOCHS
    
    print(f"[X] Scheduler Budget: {total_steps} steps.")
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=2e-4, 
        total_steps=total_steps,
        pct_start=0.1
    )
    
    scaler = torch.amp.GradScaler(device, enabled=True) if device.type == "mps" else None

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_start = time.time()
        
        total_loss = 0.0
        optimizer.zero_grad()
        
        for batch_idx, (audio, targets, diff, lengths) in enumerate(train_loader):
            audio = audio.to(device)
            targets = targets.to(device)
            diff = diff.to(device)
            lengths = lengths.to(device)
            
            context = torch.amp.autocast(device_type="mps", dtype=torch.bfloat16) if device.type == "mps" else __import__('contextlib').nullcontext()
            
            with context:
                preds = model(audio, targets, diff)
                # Phase 12b Balanced Loss
                loss, pres_l, pos_l, vel_l = compute_balanced_loss(preds, targets, lengths, gamma=1.0, pos_weight_val=12.0)
                loss = loss / ACCUM_STEPS
            
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            if (batch_idx + 1) % ACCUM_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                    
                scheduler.step()
                optimizer.zero_grad()
                
            total_loss += (loss.item() * ACCUM_STEPS)
            
            if (batch_idx + 1) % 20 == 0:
                print(f"  Batch {batch_idx+1:4d}/{len(train_loader)} | Loss: {loss.item()*ACCUM_STEPS:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        avg_loss = total_loss / len(train_loader)
        time_taken = time.time() - epoch_start
        print(f"Epoch {epoch}/{NUM_EPOCHS} | Train Loss: {avg_loss:.4f} | Time: {time_taken:.1f}s")
        
        # Validation Scan
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (audio, targets, diff, lengths) in enumerate(val_loader):
                audio = audio.to(device)
                targets = targets.to(device)
                diff = diff.to(device)
                lengths = lengths.to(device)
                with context:
                    preds = model(audio, targets, diff)
                    v_loss, _, _, _ = compute_balanced_loss(preds, targets, lengths)
                val_loss += v_loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch}/{NUM_EPOCHS} | Val Loss: {avg_val_loss:.4f}")
        
        os.makedirs("models/checkpoints", exist_ok=True)
        ckpt_path = f"models/checkpoints/transformer_phase12b_ep{epoch}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"[+] Saved checkpoint: {ckpt_path}")
        print("-" * 50)
        
    print("[X] Phase 12b Refinement Successfully Concluded.")

if __name__ == "__main__":
    run_phase12b()
