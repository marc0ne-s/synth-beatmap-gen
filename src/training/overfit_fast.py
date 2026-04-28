"""
Fast overfit test for baseline beatmap model.

Uses a tiny model and small subset for quick validation.
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.baseline import compute_loss, compute_metrics


class TinyModel(nn.Module):
    """Minimal model for fast overfit testing."""

    def __init__(self, audio_dim=80, hidden=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(audio_dim, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(hidden, hidden, num_layers=1, batch_first=True)
        self.presence_head = nn.Linear(hidden, 2)
        self.position_head = nn.Linear(hidden, 4)

    def forward(self, x):
        # x: (B, T, F)
        x = x.transpose(1, 2)  # (B, F, T)
        x = self.encoder(x)
        x = x.transpose(1, 2)  # (B, T, F)
        x, _ = self.lstm(x)
        return {
            "presence_logits": self.presence_head(x),
            "position_pred": self.position_head(x),
            "rail_logits": torch.zeros(x.shape[0], x.shape[1], 1, device=x.device),
        }


def load_features_fast(features_dir: str, num_maps: int = 10, max_frames: int = 3000):
    """Load a small subset of features into memory."""
    files = sorted([f for f in os.listdir(features_dir) if f.endswith(".npz")])[:num_maps]

    all_audio = []
    all_occ = []
    all_pos = []
    all_pres = []
    all_lengths = []

    for f in files:
        data = np.load(os.path.join(features_dir, f))
        T = min(data["note_presence"].shape[0], max_frames)

        audio = np.random.randn(T, 80).astype(np.float32)
        occ = data["note_occupancy"][:T]
        pos = data["note_positions"][:T]
        pres = data["note_presence"][:T]

        all_audio.append(audio)
        all_occ.append(occ)
        all_pos.append(pos)
        all_pres.append(pres)
        all_lengths.append(T)

    # Pad to max length
    max_len = max(all_lengths)
    B = len(files)

    audio_pad = np.zeros((B, max_len, 80), dtype=np.float32)
    occ_pad = np.zeros((B, max_len, 2, 16, 8), dtype=np.float32)
    pos_pad = np.full((B, max_len, 2, 2), -1.0, dtype=np.float32)
    pres_pad = np.zeros((B, max_len, 2), dtype=np.float32)

    for i in range(B):
        L = all_lengths[i]
        audio_pad[i, :L] = all_audio[i]
        occ_pad[i, :L] = all_occ[i]
        pos_pad[i, :L] = all_pos[i]
        pres_pad[i, :L] = all_pres[i]

    return (
        torch.from_numpy(audio_pad),
        torch.from_numpy(occ_pad),
        torch.from_numpy(pos_pad),
        torch.from_numpy(pres_pad),
        torch.tensor(all_lengths, dtype=torch.long),
    )


def train_overfit(
    features_dir: str,
    num_maps: int = 10,
    epochs: int = 50,
    lr: float = 1e-3,
    max_frames: int = 3000,
) -> dict:
    device = torch.device("cpu")
    print(f"[+] Loading {num_maps} maps, max {max_frames} frames...")

    audio, occ, pos, pres, lengths = load_features_fast(features_dir, num_maps, max_frames)
    print(f"[+] Data shape: audio={audio.shape}, pres={pres.shape}")

    model = TinyModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    audio = audio.to(device)
    pos_flat = pos.view(pos.shape[0], pos.shape[1], -1).to(device)
    pres = pres.to(device)
    lengths = lengths.to(device)
    target_rails = torch.zeros_like(pres[:, :, 0])

    print(f"[+] Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[+] Starting overfit test...")

    history = []
    for epoch in range(1, epochs + 1):
        start = time.time()

        model.train()
        optimizer.zero_grad()
        preds = model(audio)

        # Compute positive weight for class imbalance
        pos_weight = torch.tensor([
            (pres.shape[0] * pres.shape[1]) / (pres[:, :, 0].sum() + 1),
            (pres.shape[0] * pres.shape[1]) / (pres[:, :, 1].sum() + 1),
        ], device=device)

        # Weighted BCE
        presence_logits = preds["presence_logits"]
        presence_loss = nn.functional.binary_cross_entropy_with_logits(
            presence_logits, pres, pos_weight=pos_weight, reduction="mean"
        )

        # Position loss
        pos_pred = preds["position_pred"]
        pos_mask = (pres > 0.5).repeat_interleave(2, dim=-1)
        pos_loss = nn.functional.mse_loss(pos_pred * pos_mask, pos_flat * pos_mask, reduction="sum")
        pos_loss = pos_loss / (pos_mask.sum() + 1e-8)

        loss = presence_loss + pos_loss
        loss.backward()
        optimizer.step()

        loss_dict = {"total_loss": loss.item(), "presence_loss": presence_loss.item(), "position_loss": pos_loss.item()}

        model.eval()
        with torch.no_grad():
            preds = model(audio)
            metrics = compute_metrics(preds, pres, pos_flat, lengths)

        epoch_time = time.time() - start
        history.append({
            "epoch": epoch,
            "loss": loss_dict["total_loss"],
            "recall": metrics["note_recall"],
            "timing_ms": metrics["timing_error_ms"],
            "pos_err": metrics["position_error"],
        })

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:2d}/{epochs} | "
                f"loss={loss_dict['total_loss']:.4f} | "
                f"recall={metrics['note_recall']:.3f} | "
                f"timing={metrics['timing_error_ms']:.1f}ms | "
                f"pos_err={metrics['position_error']:.4f} | "
                f"time={epoch_time:.2f}s"
            )

    print("\n[+] Final Results:")
    print(f"  Note recall: {metrics['note_recall']:.3f}")
    print(f"  Timing error: {metrics['timing_error_ms']:.1f}ms")
    print(f"  Position error: {metrics['position_error']:.4f}")

    # Save
    out_dir = "models/checkpoints"
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "overfit_tiny.pt"))
    with open(os.path.join(out_dir, "overfit_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="dataset/features")
    parser.add_argument("--num-maps", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-frames", type=int, default=3000)
    args = parser.parse_args()

    train_overfit(args.features_dir, args.num_maps, args.epochs, max_frames=args.max_frames)
