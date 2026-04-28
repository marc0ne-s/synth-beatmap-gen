#!/usr/bin/env python3
"""
Training script for large beatmap model.

Same pipeline as train_baseline.py but uses LargeBeatmapModel (4M params).
Arch changes:
  - Encoder: 256 channels, 4 layers (was 128/4)
  - Decoder: 512 hidden, 4-layer LSTM (was 256/2)
  - LayerNorm on LSTM outputs
  - Precision + F1 metrics added
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.features.feature_engineering import SynthBeatmapDataset
from src.models.large_baseline import LargeBeatmapModel, compute_loss, compute_metrics


def collate_fn(batch):
    """Collate variable-length sequences."""
    audio_feats, occ, pos, pres, lengths = zip(*batch)
    audio_feats = torch.stack(audio_feats)
    occ = torch.stack(occ)
    pos = torch.stack(pos)
    pres = torch.stack(pres)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return audio_feats, occ, pos, pres, lengths


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> dict:
    model.train()
    total_loss = 0.0
    total_presence = 0.0
    total_position = 0.0
    total_rail = 0.0
    num_batches = 0

    for batch in dataloader:
        audio_feats, occ, pos, pres, lengths = [b.to(device) for b in batch]

        optimizer.zero_grad()
        predictions = model(audio_feats)

        target_presence = pres
        target_positions = pos.view(pos.shape[0], pos.shape[1], -1)
        target_rails = torch.zeros_like(pres[:, :, 0])

        # Per-hand positive weight for class imbalance (~5% positive)
        pos_weight = torch.tensor([
            (target_presence.shape[0] * target_presence.shape[1]) / (target_presence[:, :, 0].sum() + 1),
            (target_presence.shape[0] * target_presence.shape[1]) / (target_presence[:, :, 1].sum() + 1),
        ], device=device)

        loss, loss_dict = compute_loss(
            predictions,
            target_presence,
            target_positions,
            target_rails,
            lengths,
            pos_weight=pos_weight,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss_dict["total_loss"]
        total_presence += loss_dict["presence_loss"]
        total_position += loss_dict["position_loss"]
        total_rail += loss_dict["rail_loss"]
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "presence_loss": total_presence / num_batches,
        "position_loss": total_position / num_batches,
        "rail_loss": total_rail / num_batches,
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_recall = 0.0
    total_precision = 0.0
    total_f1 = 0.0
    total_timing = 0.0
    total_pos_err = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            audio_feats, occ, pos, pres, lengths = [b.to(device) for b in batch]
            predictions = model(audio_feats)

            target_presence = pres
            target_positions = pos.view(pos.shape[0], pos.shape[1], -1)
            target_rails = torch.zeros_like(pres[:, :, 0])

            pos_weight_val = torch.tensor([
                (target_presence.shape[0] * target_presence.shape[1]) / (target_presence[:, :, 0].sum() + 1),
                (target_presence.shape[0] * target_presence.shape[1]) / (target_presence[:, :, 1].sum() + 1),
            ], device=device)

            loss, _ = compute_loss(
                predictions,
                target_presence,
                target_positions,
                target_rails,
                lengths,
                pos_weight=pos_weight_val,
            )

            metrics = compute_metrics(
                predictions,
                target_presence,
                target_positions,
                lengths,
            )

            total_loss += loss.item()
            total_recall += metrics["note_recall"]
            total_precision += metrics["precision"]
            total_f1 += metrics["f1"]
            total_timing += metrics["timing_error_ms"]
            total_pos_err += metrics["position_error"]
            num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "note_recall": total_recall / num_batches,
        "precision": total_precision / num_batches,
        "f1": total_f1 / num_batches,
        "timing_error_ms": total_timing / num_batches,
        "position_error": total_pos_err / num_batches,
    }


def train(
    features_dir: str,
    output_dir: str,
    difficulty: str = "Hard",
    num_maps: int = 100,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
    device_str: str = "mps",
    audio_features_dir: Optional[str] = None,
    train_split: float = 0.8,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(device_str if torch.backends.mps.is_available() else "cpu")
    print(f"[+] Using device: {device}")

    # Load dataset
    print(f"[+] Loading dataset from {features_dir}")
    dataset = SynthBeatmapDataset(features_dir, difficulty=difficulty, audio_features_dir=audio_features_dir)

    if len(dataset) == 0:
        raise ValueError("No valid feature files found!")

    # Subset
    num_maps = min(num_maps, len(dataset))
    subset = Subset(dataset, range(num_maps))
    print(f"[+] Training set: {num_maps} maps")

    # Split train/val with deterministic shuffle (seed=42)
    indices = list(range(len(subset)))
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    split_idx = int(len(indices) * train_split)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_subset = Subset(subset, train_indices)
    val_subset = Subset(subset, val_indices)
    print(f"[+] Train/val split: {len(train_subset)}/{len(val_subset)} maps")

    # DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Model
    model = LargeBeatmapModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    print(f"[+] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    history = []
    best_f1 = 0.0

    for epoch in range(1, epochs + 1):
        start = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["loss"])

        epoch_time = time.time() - start
        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "time": epoch_time,
        })

        # Save best by F1 (not loss — silence has low loss!)
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            ckpt_path = os.path.join(output_dir, "large_best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "f1": best_f1,
            }, ckpt_path)

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"recall={val_metrics['note_recall']:.3f} | "
            f"precision={val_metrics['precision']:.3f} | "
            f"f1={val_metrics['f1']:.3f} | "
            f"timing_err={val_metrics['timing_error_ms']:.1f}ms | "
            f"pos_err={val_metrics['position_error']:.4f} | "
            f"time={epoch_time:.1f}s",
            flush=True,
        )

    # Save history
    history_path = os.path.join(output_dir, "large_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[+] Saved training history to {history_path}")

    # Final metrics
    print("\n[+] Final metrics:")
    print(f"  Note recall: {val_metrics['note_recall']:.3f}")
    print(f"  Precision: {val_metrics['precision']:.3f}")
    print(f"  F1 Score: {val_metrics['f1']:.3f}")
    print(f"  Timing error: {val_metrics['timing_error_ms']:.1f}ms")
    print(f"  Position error: {val_metrics['position_error']:.4f}")
    print(f"  Best val F1: {best_f1:.4f}")

    return history


def main():
    parser = argparse.ArgumentParser(description="Train large beatmap model")
    parser.add_argument("--features-dir", default="/Volumes/Second-Brain-1/AI/Synth/dataset/features")
    parser.add_argument("--output-dir", default="/Volumes/Second-Brain-1/AI/Synth/models/checkpoints")
    parser.add_argument("--audio-features-dir", default=None, help="Directory with real audio mel .npz files")
    parser.add_argument("--difficulty", default="Hard")
    parser.add_argument("--num-maps", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--train-split", type=float, default=0.8, help="Fraction of num_maps for training (rest = validation)")
    args = parser.parse_args()

    train(
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        difficulty=args.difficulty,
        num_maps=args.num_maps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_str=args.device,
        audio_features_dir=args.audio_features_dir,
        train_split=args.train_split,
    )


if __name__ == "__main__":
    main()
