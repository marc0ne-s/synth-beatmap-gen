import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.streaming_loader import get_streaming_loader
from src.models.hybrid_generator import HybridCascadeModel
from src.training.train_transformer_pilot import compute_masked_loss

def run_hybrid_training():
    print("=====================================================")
    print("  PHASE 11: HYBRID CASCADE TRAINING")
    print("=====================================================")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[X] Targeting Accelerator: {device}")
    
    features_dir = "/Volumes/Second-Brain-1/AI/Synth/dataset/features"
    audio_dir = "/Volumes/Second-Brain-1/AI/Synth/dataset/audio_features"
    
    BATCH_SIZE = 8
    ACCUM_STEPS = 4
    NUM_EPOCHS = 5
    
    train_loader, val_loader = get_streaming_loader(
        features_dir=features_dir,
        audio_dir=audio_dir,
        difficulties=["Hard", "Expert", "Master"],
        batch_size=BATCH_SIZE,
        num_maps=2500
    )
    
    print("[X] Initializing Hybrid Cascade...")
    model = HybridCascadeModel().to(device)
    
    # 1. Load LSTM & Phase 10 Transformer Memory
    model.load_states(
        baseline_ckpt="models/checkpoints/best_model.pt",
        transformer_ckpt="models/checkpoints/transformer_pilot_ep5.pt",
        device=device
    )
    
    # 1.5 Phase 11: RESET Heads to purge Phase 10 "Shatter" weights
    print("[X] Resetting Transformer Output Heads...")
    nn.init.kaiming_uniform_(model.transformer.presence_out.weight, a=math.sqrt(5))
    nn.init.kaiming_uniform_(model.transformer.position_out.weight, a=math.sqrt(5))
    nn.init.kaiming_uniform_(model.transformer.velocity_out.weight, a=math.sqrt(5))
    nn.init.zeros_(model.transformer.presence_out.bias)
    nn.init.zeros_(model.transformer.position_out.bias)
    nn.init.zeros_(model.transformer.velocity_out.bias)
    
    # 2. Split Learning Rate Formulation
    # coarse_proj and Heads learn fast (5e-4),
    # while the rest of the transformer gently fine-tunes (1e-4).
    fast_params = []
    gentle_params = []
    
    for n, p in model.transformer.named_parameters():
        if "coarse_proj" in n or "out" in n or "proj" in n: # Let projections and heads learn fast
            fast_params.append(p)
        else:
            gentle_params.append(p)
            
    optimizer = optim.AdamW([
        {"params": fast_params, "lr": 5e-4},
        {"params": gentle_params, "lr": 1e-4}
    ], weight_decay=0.01)
    
    total_steps_per_epoch = (len(train_loader) + ACCUM_STEPS - 1) // ACCUM_STEPS
    
    # We no longer use OneCycleLR since we start from highly trained distributions. 
    # Let's use CosineAnnealingLR for gentle finish.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS * total_steps_per_epoch, eta_min=1e-5)
    
    scaler = torch.amp.GradScaler(device, enabled=True) if device.type == "mps" else None

    # Proceed to train purely the Transformer within the hybrid.
    for epoch in range(1, NUM_EPOCHS + 1):
        model.transformer.train()
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
                # model.forward() internally runs baseline -> transformer
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
                torch.nn.utils.clip_grad_norm_(model.transformer.parameters(), max_norm=1.0)
                
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
        
        model.transformer.eval()
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
        ckpt_path = f"models/checkpoints/hybrid_cascade_ep{epoch}.pt"
        torch.save(model.transformer.state_dict(), ckpt_path)
        print(f"[+] Saved checkpoint: {ckpt_path}")
        print("-" * 50)
        
    print("[X] Phase 11 Cascade Training Concluded.")

if __name__ == "__main__":
    run_hybrid_training()
