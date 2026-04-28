"""
Autoresearch training script for SynthRiders beatmap generation.

This is the only file the AI agent should modify.
Usage: python autoresearch/synth_train.py
"""

import os
import sys
import time
import json
import math
from pathlib import Path
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.streaming_loader import get_streaming_loader
from src.models.baseline import BaselineBeatmapModel
from src.models.transformer import TransformerCausalDecoder

# ---------------------------------------------------------------------------
# Experiment Config
# ---------------------------------------------------------------------------

@dataclass
class SynthConfig:
    d_model: int = 256
    num_layers: int = 4
    d_audio: int = 128
    d_target: int = 8
    num_diff: int = 6
    batch_size: int = 8
    accum_steps: int = 4
    num_epochs: int = 5
    max_lr: float = 5e-4
    base_lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    num_maps: int = 2500
    difficulties: list = None
    features_dir: str = "dataset/features"
    audio_dir: str = "dataset/audio_features"
    transformer_ckpt: str = "models/checkpoints/transformer_pilot_ep5.pt"
    baseline_ckpt: str = "models/checkpoints/best_model.pt"
    save_dir: str = "models/checkpoints"
    run_tag: str = "autoresearch"
    use_baseline: bool = True
    freeze_baseline: bool = True

    def __post_init__(self):
        if self.difficulties is None:
            self.difficulties = ["Hard", "Expert", "Master"]


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

def compute_masked_loss(predictions, target_features, lengths, gamma=2.0, pos_weight=1.5):
    pres_logits = predictions["presence_logits"]
    pos_pred = predictions["position_pred"]
    vel_pred = predictions["velocity_pred"]
    
    target_pres = target_features[..., 0:2]
    target_pos = target_features[..., 2:6]
    target_vel = target_features[..., 6:8]
    
    B, T, _ = target_pres.shape
    device = pres_logits.device
    mask = torch.arange(T, device=device).unsqueeze(0).expand(B, T) < lengths.unsqueeze(1)
    
    pw = torch.tensor([pos_weight, pos_weight], device=device)
    bce = F.binary_cross_entropy_with_logits(pres_logits, target_pres, pos_weight=pw, reduction='none')
    p = torch.sigmoid(pres_logits)
    p_t = p * target_pres + (1 - p) * (1 - target_pres)
    focal_weight = (1 - p_t) ** gamma
    focal_loss = (focal_weight * bce) * mask.unsqueeze(-1)
    loss_pres = focal_loss.sum() / (mask.sum() * 2 + 1e-8)
    
    active_mask = mask.unsqueeze(-1) & (target_pres > 0.5)
    active_pos = torch.cat([
        active_mask[..., 0:1], active_mask[..., 0:1],
        active_mask[..., 1:2], active_mask[..., 1:2]
    ], dim=-1)
    se_pos = F.smooth_l1_loss(pos_pred, target_pos, reduction='none')
    se_pos = se_pos * active_pos
    loss_pos = se_pos.sum() / (active_pos.sum() + 1e-8)
    
    se_vel = F.smooth_l1_loss(vel_pred, target_vel, reduction='none')
    se_vel = se_vel * active_mask
    loss_vel = se_vel.sum() / (active_mask.sum() + 1e-8)
    
    total = loss_pres + (loss_pos * 10.0) + (loss_vel * 25.0)
    return total, loss_pres, loss_pos, loss_vel


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_recall_at_50ms(pred_pres, gt_pres, threshold_frames=2.5):
    recalls = []
    for hand in range(2):
        pred_times = pred_pres[:, hand].nonzero(as_tuple=True)[0]
        gt_times = gt_pres[:, hand].nonzero(as_tuple=True)[0]
        if len(gt_times) == 0:
            continue
        matched = 0
        for gt_t in gt_times:
            if len(pred_times) > 0:
                dists = torch.abs(pred_times.float() - gt_t.float())
                if dists.min() <= threshold_frames:
                    matched += 1
        recalls.append(matched / len(gt_times))
    return sum(recalls) / len(recalls) if recalls else 0.0


def evaluate_batch(preds, targets, lengths):
    pred_pres = torch.sigmoid(preds["presence_logits"]) > 0.5
    true_pres = targets[..., 0:2] > 0.5
    
    recalls = []
    precisions = []
    B = lengths.shape[0]
    
    for b in range(B):
        T_valid = lengths[b].item()
        r = compute_recall_at_50ms(pred_pres[b, :T_valid], true_pres[b, :T_valid])
        p = compute_recall_at_50ms(true_pres[b, :T_valid], pred_pres[b, :T_valid])
        recalls.append(r)
        precisions.append(p)
    
    return sum(recalls) / B, sum(precisions) / B


def compute_metrics(model, loader, device, baseline=None, use_baseline=True):
    model.eval()
    if baseline is not None:
        baseline.eval()
    
    total_loss = 0.0
    total_pres = 0.0
    total_pos = 0.0
    total_vel = 0.0
    total_recall = 0.0
    total_precision = 0.0
    num_batches = 0
    
    context = torch.amp.autocast(device_type="mps", dtype=torch.bfloat16) if device.type == "mps" else __import__('contextlib').nullcontext()
    
    with torch.no_grad():
        for audio, targets, diff, lengths in loader:
            audio = audio.to(device)
            targets = targets.to(device)
            diff = diff.to(device)
            lengths = lengths.to(device)
            
            with context:
                if baseline is not None and use_baseline:
                    with torch.no_grad():
                        coarse_out = baseline(audio[..., :80])
                        coarse_memory = coarse_out.get("lstm_out", None)
                    preds = model(audio, targets, diff, coarse_memory=coarse_memory)
                else:
                    preds = model(audio, targets, diff)
                
                loss, pres_l, pos_l, vel_l = compute_masked_loss(preds, targets, lengths)
            
            recall, precision = evaluate_batch(preds, targets, lengths)
            
            total_loss += loss.item()
            total_pres += pres_l.item()
            total_pos += pos_l.item()
            total_vel += vel_l.item()
            total_recall += recall
            total_precision += precision
            num_batches += 1
    
    return {
        "val_loss": total_loss / num_batches,
        "val_presence": total_pres / num_batches,
        "val_position": total_pos / num_batches,
        "val_velocity": total_vel / num_batches,
        "val_recall_50": total_recall / num_batches,
        "val_precision_50": total_precision / num_batches,
    }


# ---------------------------------------------------------------------------
# Coarse Fusion Wrapper (for baseline hybrid training)
# ---------------------------------------------------------------------------

class CoarseFusionWrapper(nn.Module):
    """Wrap TransformerCausalDecoder and add coarse_memory fusion."""
    def __init__(self, transformer, d_model, d_coarse=256):
        super().__init__()
        self.transformer = transformer
        self.coarse_proj = nn.Linear(d_coarse, d_model)

    def forward(self, audio_features, target_features, difficulty_idx, coarse_memory=None):
        x_audio = self.transformer.pos_enc(self.transformer.audio_proj(audio_features))
        shifted_targets = torch.zeros_like(target_features)
        shifted_targets[:, 1:, :] = target_features[:, :-1, :]
        v_tgt = self.transformer.pos_enc(self.transformer.target_proj(shifted_targets))
        v_diff = self.transformer.diff_emb(difficulty_idx).unsqueeze(1).expand(-1, target_features.size(1), -1)
        x = v_tgt + x_audio + v_diff
        if coarse_memory is not None:
            x = x + self.coarse_proj(coarse_memory)
        for layer in self.transformer.layers:
            x = layer(x, memory_audio=x_audio)
        presence_logits = self.transformer.presence_out(x)
        position_pred = torch.tanh(self.transformer.position_out(x))
        velocity_pred = torch.tanh(self.transformer.velocity_out(x))
        return {
            "presence_logits": presence_logits,
            "position_pred": position_pred,
            "velocity_pred": velocity_pred,
            "latent_state": x,
        }


# ---------------------------------------------------------------------------
# Model Builder
# ---------------------------------------------------------------------------

def build_hybrid_model(config, device):
    transformer = TransformerCausalDecoder(
        d_model=config.d_model,
        num_layers=config.num_layers,
        d_audio=config.d_audio,
        d_target=config.d_target,
        num_diff=config.num_diff
    ).to(device)
    
    ckpt_path = PROJECT_ROOT / config.transformer_ckpt
    if ckpt_path.exists():
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        model_state = transformer.state_dict()
        filtered_state = {}
        for k, v in state_dict.items():
            if k in model_state and v.shape == model_state[k].shape:
                filtered_state[k] = v
            else:
                if k in model_state:
                    print(f"[!] Skipping {k}: checkpoint {tuple(v.shape)} vs model {tuple(model_state[k].shape)}")
                else:
                    print(f"[!] Skipping unknown key: {k}")
        transformer.load_state_dict(filtered_state, strict=False)
        print(f"[+] Loaded transformer from {ckpt_path}")
    else:
        print(f"[!] No checkpoint at {ckpt_path}, using fresh init")
    
    baseline = None
    if config.use_baseline:
        baseline_ckpt = PROJECT_ROOT / config.baseline_ckpt
        if baseline_ckpt.exists():
            try:
                baseline = BaselineBeatmapModel()
                ckpt = torch.load(baseline_ckpt, map_location=device, weights_only=True)
                # Handle both raw state_dict and full checkpoint dicts
                sd = ckpt.get("model_state_dict", ckpt)
                baseline.load_state_dict(sd, strict=False)
                if config.freeze_baseline:
                    baseline.requires_grad_(False)
                baseline = baseline.to(device)
                print(f"[+] Loaded baseline from {baseline_ckpt}")
            except Exception as e:
                print(f"[!] Could not load baseline: {e}")
                baseline = None
        else:
            print(f"[!] No baseline at {baseline_ckpt}")
    
    # Wrap transformer for coarse_memory fusion
    model = CoarseFusionWrapper(transformer, config.d_model, d_coarse=256)
    model = model.to(device)
    return model, baseline


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def run_training(config):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[X] Accelerator: {device}")
    
    features_dir = str(PROJECT_ROOT / config.features_dir)
    audio_dir = str(PROJECT_ROOT / config.audio_dir)
    
    train_loader, val_loader = get_streaming_loader(
        features_dir=features_dir,
        audio_dir=audio_dir,
        difficulties=config.difficulties,
        batch_size=config.batch_size,
        num_maps=config.num_maps,
    )
    
    model, baseline = build_hybrid_model(config, device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[X] Transformer params: {total_params:,} (trainable: {trainable_params:,})")
    
    # Optimizer
    coarse_proj_params = list(model.coarse_proj.parameters())
    other_params = [p for n, p in model.named_parameters() if "coarse_proj" not in n]
    
    optimizer = optim.AdamW([
        {"params": coarse_proj_params, "lr": config.max_lr},
        {"params": other_params, "lr": config.base_lr},
    ], weight_decay=config.weight_decay)
    
    total_steps = (len(train_loader) + config.accum_steps - 1) // config.accum_steps
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[config.max_lr, config.base_lr * 5],
        epochs=config.num_epochs,
        steps_per_epoch=total_steps,
        pct_start=0.3,
    )
    
    scaler = torch.amp.GradScaler(device, enabled=True) if device.type == "mps" else None
    context = torch.amp.autocast(device_type="mps", dtype=torch.bfloat16) if device.type == "mps" else __import__('contextlib').nullcontext()
    
    best_score = -float('inf')
    best_epoch = 0
    history = []
    
    for epoch in range(1, config.num_epochs + 1):
        epoch_start = time.time()
        model.train()
        
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        for batch_idx, (audio, targets, diff, lengths) in enumerate(train_loader):
            audio = audio.to(device)
            targets = targets.to(device)
            diff = diff.to(device)
            lengths = lengths.to(device)
            
            with context:
                if baseline is not None and config.use_baseline:
                    with torch.no_grad():
                        coarse_out = baseline(audio[..., :80])
                        coarse_memory = coarse_out.get("lstm_out", None)
                    preds = model(audio, targets, diff, coarse_memory=coarse_memory)
                else:
                    preds = model(audio, targets, diff)
                
                loss, pres_l, pos_l, vel_l = compute_masked_loss(preds, targets, lengths)
                loss = loss / config.accum_steps
            
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (batch_idx + 1) % config.accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
            
            epoch_loss += (loss.item() * config.accum_steps)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1:3d}/{len(train_loader)} | Loss: {loss.item()*config.accum_steps:.4f}")
        
        avg_train_loss = epoch_loss / len(train_loader)
        
        metrics = compute_metrics(model, val_loader, device, baseline, use_baseline=config.use_baseline)
        metrics["train_loss"] = avg_train_loss
        metrics["epoch"] = epoch
        metrics["time"] = time.time() - epoch_start
        
        score = metrics["val_recall_50"] * (1.0 / (1.0 + metrics["val_position"]))
        metrics["score"] = score
        history.append(metrics)
        
        print(f"Epoch {epoch}/{config.num_epochs} | "
              f"Train: {avg_train_loss:.4f} | "
              f"Val Recall@50: {metrics['val_recall_50']:.2%} | "
              f"Val Pos MSE: {metrics['val_position']:.4f} | "
              f"Score: {score:.4f} | "
              f"Time: {metrics['time']:.1f}s")
        
        if score > best_score:
            best_score = score
            best_epoch = epoch
            save_p = Path(config.save_dir) / f"autoresearch_{config.run_tag}_best.pt"
            save_p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_p)
            print(f"[+] NEW BEST (score={score:.4f}) → {save_p}")
        
        print("-" * 60)
    
    summary = {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "final_metrics": history[-1],
        "config": asdict(config),
    }
    
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config = SynthConfig(
        run_tag=os.getenv("AUTORESEARCH_TAG", "default"),
        num_epochs=int(os.getenv("AUTORESEARCH_EPOCHS", "5")),
    )
    
    # Quick dry-run to check if baseline loads etc.
    print("=" * 50)
    print("SYNTH AUTORESEARCH — Baseline Experiment")
    print("=" * 50)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checking data availability...")
    
    # Check data
    feat_dir = PROJECT_ROOT / config.features_dir
    aud_dir = PROJECT_ROOT / config.audio_dir
    feat_files = list(feat_dir.glob("*.npz"))[:5]
    aud_files = list(aud_dir.glob("*.npz"))[:5]
    
    print(f"Feature files (sample): {[f.name for f in feat_files]}")
    print(f"Audio files (sample): {[f.name for f in aud_files]}")
    print(f"Data check: {'PASS' if feat_files and aud_files else 'FAIL'}")
    
    if not feat_files or not aud_files:
        print("ERROR: No data files found. Cannot proceed.")
        return {"status": "aborted", "reason": "no_data"}
    
    # Run training
    summary = run_training(config)
    
    # Calculate training time
    total_time = sum(m["time"] for m in summary.get("history", []))
    
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"val_recall_50:    {summary['final_metrics']['val_recall_50']:.6f}")
    print(f"val_position:     {summary['final_metrics']['val_position']:.6f}")
    print(f"val_precision_50: {summary['final_metrics']['val_precision_50']:.6f}")
    print(f"val_loss:         {summary['final_metrics']['val_loss']:.6f}")
    print(f"score:            {summary['best_score']:.6f}")
    print(f"training_seconds: {total_time:.1f}")
    print(f"peak_vram_mb:     {torch.mps.current_allocated_memory() / 1024**2 if torch.backends.mps.is_available() else 0:.1f}")
    print("=" * 50)
    
    summary["status"] = "complete"
    return summary


if __name__ == "__main__":
    summary = main()
