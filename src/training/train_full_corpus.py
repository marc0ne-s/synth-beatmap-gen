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
from src.training.train_transformer_pilot import compute_masked_loss

def run_phase12():
    print("=====================================================")
    print("  PHASE 12: FULL CORPUS TRANSFORMER SCALE-UP")
    print("=====================================================")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[X] Targeting Accelerator: {device}")
    
    features_dir = "/Volumes/Second-Brain-1/AI/Synth/dataset/features"
    audio_dir = "/Volumes/Second-Brain-1/AI/Synth/dataset/audio_features"
    
    BATCH_SIZE = 8
    ACCUM_STEPS = 8  # Effective Batch Size 64 for stability at scale
    NUM_EPOCHS = 10  # Full convergence run
    
    train_loader, val_loader = get_streaming_loader(
        features_dir=features_dir,
        audio_dir=audio_dir,
        difficulties=["Hard", "Expert", "Master"],
        batch_size=BATCH_SIZE,
        num_maps=8861  # FULL CORPUS
    )
    
    model = TransformerCausalDecoder(
        d_model=256,
        num_layers=4,
        d_audio=128,
        d_target=8
    ).to(device)
    
    # 1. Load Acoustic Memory from Phase 10
    print("[X] Loading Acoustic Anchors from Phase 10 (ep5)...")
    ckpt_path = "models/checkpoints/transformer_pilot_ep5.pt"
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        print("[!] Phase 10 checkpoint not found. Starting from scratch.")

    # 2. Hard Reset Output Heads (Purge Phase 10 spatial breakdown)
    print("[X] Resetting Spatial Mapping Heads...")
    nn.init.kaiming_uniform_(model.presence_out.weight, a=math.sqrt(5))
    nn.init.kaiming_uniform_(model.position_out.weight, a=math.sqrt(5))
    nn.init.kaiming_uniform_(model.velocity_out.weight, a=math.sqrt(5))
    nn.init.zeros_(model.presence_out.bias)
    nn.init.zeros_(model.position_out.bias)
    nn.init.zeros_(model.velocity_out.bias)
    
    # 3. Discriminative LR: Let heads learn fast, anchors learn slow
    head_params = list(model.presence_out.parameters()) + \
                  list(model.position_out.parameters()) + \
                  list(model.velocity_out.parameters())
    base_params = [p for p in model.parameters() if not any(p is hp for hp in head_params)]
    
    optimizer = optim.AdamW([
        {"params": head_params, "lr": 5e-4}, 
        {"params": base_params, "lr": 1e-4}
    ], weight_decay=0.01)
    
    total_steps = len(train_loader) // ACCUM_STEPS * NUM_EPOCHS
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=[8e-4, 2e-4], # Higher peak for heads
        total_steps=total_steps,
        pct_start=0.2
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
                loss, pres_l, pos_l, vel_l = compute_masked_loss(preds, targets, lengths)
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
                print(f"  Batch {batch_idx+1:4d}/{len(train_loader)} | Loss: {loss.item()*ACCUM_STEPS:.4f}")

        avg_loss = total_loss / len(train_loader)
        time_taken = time.time() - epoch_start
        print(f"Epoch {epoch}/{NUM_EPOCHS} | Train Loss: {avg_loss:.4f} | Time: {time_taken:.1f}s")
        
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
        
        os.makedirs("models/checkpoints", exist_ok=True)
        ckpt_path = f"models/checkpoints/transformer_full_ep{epoch}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"[+] Saved checkpoint: {ckpt_path}")
        print("-" * 50)
        
    print("[X] Phase 12 Training Successfully Concluded.")

if __name__ == "__main__":
    run_phase12()
