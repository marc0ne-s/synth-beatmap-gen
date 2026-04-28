"""
Architect's Pilot: The Multi-Difficulty, MPS-Native Validation Run.
Tests the TransformerCausalDecoder on an 800-map mixed-difficulty stream.
"""

import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Ensure paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.streaming_loader import get_streaming_loader
from src.models.transformer import TransformerCausalDecoder

def compute_masked_loss(predictions, target_features, lengths):
    """
    Computes masked BCE for presence, and masked MSE (Huber) for positions and velocity.
    target_features: (B, T, 8) -> pres=(0:2), pos=(2:6), vel=(6:8)
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
    
    # 1. Presence Loss (Focal Loss)
    pos_weight = torch.tensor([1.5, 1.5], device=device) # Surgical Precision rebalance
    bce = F.binary_cross_entropy_with_logits(pres_logits, target_pres, pos_weight=pos_weight, reduction='none')
    
    # Probability extraction for Focal weighting
    p = torch.sigmoid(pres_logits)
    p_t = p * target_pres + (1 - p) * (1 - target_pres)
    
    # Focal Suppression Formulation (Gamma=2.0)
    focal_weight = (1 - p_t) ** 2.0
    focal_loss = focal_weight * bce
    
    focal_loss = focal_loss * mask.unsqueeze(-1)
    loss_pres = focal_loss.sum() / (mask.sum() * 2 + 1e-8)
    
    # 2. Position Loss (Huber Loss)
    active_mask = mask.unsqueeze(-1) & (target_pres > 0.5)
    se_pos = F.smooth_l1_loss(pos_pred, target_pos, reduction='none')
    active_mask_pos = torch.cat([active_mask[..., 0:1], active_mask[..., 0:1], 
                                 active_mask[..., 1:2], active_mask[..., 1:2]], dim=-1)
    se_pos = se_pos * active_mask_pos
    loss_pos = se_pos.sum() / (active_mask_pos.sum() + 1e-8)
    
    # 3. Velocity / Intensity Loss (Huber Loss)
    se_vel = F.smooth_l1_loss(vel_pred, target_vel, reduction='none')
    # Target_vel and active_mask are precisely (B, T, 2) handling Right/Left Hands identically
    se_vel = se_vel * active_mask
    loss_vel = se_vel.sum() / (active_mask.sum() + 1e-8)
    
    # Balance magnitudes: Velocity delta is intrinsically minute (often 0.1 - 0.5 range), 
    # so we multiply by 25.0 to give it potent gradient visibility against Presence.
    total = loss_pres + (loss_pos * 10.0) + (loss_vel * 25.0)
    
    return total, loss_pres, loss_pos, loss_vel

def run_pilot():
    print("=====================================================")
    print("  PHASE 3 PILOT: M4 TRANSFORMER & STREAMING DATA")
    print("=====================================================")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[X] Targeting Accelerator: {device}")
    
    # Paths
    features_dir = "/Volumes/Second-Brain-1/AI/Synth/dataset/features"
    audio_dir = "/Volumes/Second-Brain-1/AI/Synth/dataset/audio_features"
    
    # 1. Initialize M4-Native Stream
    BATCH_SIZE = 8
    ACCUM_STEPS = 4  # Simulate Batch Size 32
    NUM_EPOCHS = 5
    
    train_loader, val_loader = get_streaming_loader(
        features_dir=features_dir,
        audio_dir=audio_dir,
        difficulties=["Hard", "Expert", "Master"],
        batch_size=BATCH_SIZE,
        num_maps=2500  # The Great Expansion
    )
    
    # 2. Initialize Model
    model = TransformerCausalDecoder(
        d_model=256,
        num_layers=4,
        d_audio=128,
        d_target=8
    ).to(device)
    
    print(f"[X] Transformer Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # OneCycleLR Scheduler (Scaling the 2,500 map sequence)
    total_steps_per_epoch = (len(train_loader) + ACCUM_STEPS - 1) // ACCUM_STEPS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=5e-4, 
        epochs=NUM_EPOCHS, 
        steps_per_epoch=total_steps_per_epoch
    )
    
    # 3. MPS Validated Training Loop
    scaler = torch.amp.GradScaler(device, enabled=True) if device.type == "mps" else None

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_start = time.time()
        
        total_loss = 0.0
        optimizer.zero_grad()
        
        for batch_idx, (audio, targets, diff, lengths) in enumerate(train_loader):
            # Late-device mapping
            audio = audio.to(device)
            targets = targets.to(device)
            diff = diff.to(device)
            lengths = lengths.to(device)
            
            # Autocast blocks float16 execution inherently on MPS
            context = torch.amp.autocast(device_type="mps", dtype=torch.bfloat16) if device.type == "mps" else __import__('contextlib').nullcontext()
            
            with context:
                preds = model(audio, targets, diff)
                loss, pres_l, pos_l, vel_l = compute_masked_loss(preds, targets, lengths)
                loss = loss / ACCUM_STEPS
            
            # Backward
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            # Accumulate
            if (batch_idx + 1) % ACCUM_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                # Gradient Clipping is essential for Transformers
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
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1:3d}/{len(train_loader)} | Cur Loss: {loss.item()*ACCUM_STEPS:.4f}")

        avg_loss = total_loss / len(train_loader)
        time_taken = time.time() - epoch_start
        print(f"Epoch {epoch}/{NUM_EPOCHS} | Train Loss: {avg_loss:.4f} | Time: {time_taken:.1f}s")
        
        # Super lightweight validation scan
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
                    v_loss, _, _, _ = compute_masked_loss(preds, targets, lengths)
                val_loss += v_loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch}/{NUM_EPOCHS} | Val Loss: {avg_val_loss:.4f}")
        
        # Save Pilot Checkpoint
        os.makedirs("models/checkpoints", exist_ok=True)
        ckpt_path = f"models/checkpoints/transformer_pilot_ep{epoch}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"[+] Saved checkpoint: {ckpt_path}")
        print("-" * 50)
        
    print("[X] Pilot Successfully Concluded.")

if __name__ == "__main__":
    run_pilot()
