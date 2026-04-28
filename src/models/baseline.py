"""
Baseline beatmap prediction model for SynthRiders.

Architecture:
    Audio Features (T, F)
      -> Conv1D Encoder
      -> Latent sequence (T, D)
      -> LSTM Decoder (causal)
      -> Per-frame predictions:
           - note presence (2 hands, binary)
           - note positions (2 hands, x/y regression)
           - rail flag (binary)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple  # Added for Python 3.9 compatibility


class Conv1DEncoder(nn.Module):
    """Encode audio features into a latent sequence."""

    def __init__(
        self,
        in_channels: int = 80,
        hidden_channels: int = 128,
        num_layers: int = 4,
        kernel_size: int = 3,
    ):
        super().__init__()
        layers = []
        channels = [in_channels] + [hidden_channels] * num_layers
        for i in range(num_layers):
            layers.extend([
                nn.Conv1d(channels[i], channels[i + 1], kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(channels[i + 1]),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
        self.encoder = nn.Sequential(*layers)
        self.out_channels = hidden_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F)
        Returns:
            (B, T, D)
        """
        x = x.transpose(1, 2)  # (B, F, T)
        x = self.encoder(x)
        x = x.transpose(1, 2)  # (B, T, D)
        return x


class BeatmapDecoder(nn.Module):
    """Decode latent sequence into per-frame beatmap predictions."""

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        x_bins: int = 16,
        y_bins: int = 8,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )

        # Multi-task heads
        self.presence_head = nn.Linear(hidden_dim, 2)  # binary per hand
        # Initialize presence bias to match ~16% base rate (logit ≈ -1.6)
        # Prevents focal loss from collapsing to all-silence or all-notes at start
        nn.init.constant_(self.presence_head.bias, -1.6)
        self.position_head = nn.Linear(hidden_dim, 4)   # x, y per hand
        self.rail_head = nn.Linear(hidden_dim, 1)       # rail flag

        self.x_bins = x_bins
        self.y_bins = y_bins

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (B, T, D) latent sequence
        Returns:
            dict with keys:
                presence_logits: (B, T, 2)
                position_pred: (B, T, 4)  # [right_x, right_y, left_x, left_y]
                rail_logits: (B, T, 1)
        """
        # Causal LSTM
        out, _ = self.lstm(x)

        presence_logits = self.presence_head(out)
        position_pred = self.position_head(out)
        rail_logits = self.rail_head(out)

        return {
            "presence_logits": presence_logits,
            "position_pred": position_pred,
            "rail_logits": rail_logits,
            "lstm_out": out,
        }


class BaselineBeatmapModel(nn.Module):
    """End-to-end beatmap prediction model."""

    def __init__(
        self,
        audio_features: int = 80,
        encoder_hidden: int = 128,
        encoder_layers: int = 4,
        decoder_hidden: int = 256,
        decoder_layers: int = 2,
    ):
        super().__init__()
        self.encoder = Conv1DEncoder(
            in_channels=audio_features,
            hidden_channels=encoder_hidden,
            num_layers=encoder_layers,
        )
        self.decoder = BeatmapDecoder(
            latent_dim=encoder_hidden,
            hidden_dim=decoder_hidden,
            num_layers=decoder_layers,
        )

    def forward(self, audio_features: torch.Tensor) -> dict:
        """
        Args:
            audio_features: (B, T, F)
        Returns:
            dict with presence_logits, position_pred, rail_logits
        """
        latent = self.encoder(audio_features)
        return self.decoder(latent)


def compute_loss(
    predictions: dict,
    target_presence: torch.Tensor,
    target_positions: torch.Tensor,
    target_rails: torch.Tensor,
    lengths: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute multi-task loss.

    Args:
        predictions: model output dict
        target_presence: (B, T, 2) binary
        target_positions: (B, T, 4) continuous, -1 for no note
        target_rails: (B, T) binary
        lengths: (B,) actual sequence lengths
        pos_weight: (2,) per-hand positive weight for class imbalance

    Returns:
        total_loss, loss_dict
    """
    B, T = target_presence.shape[:2]

    # Create mask for valid frames
    mask = torch.arange(T, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1)  # (B, T, 1)

    # Presence loss (BCE) with optional pos_weight for class imbalance
    presence_logits = predictions["presence_logits"]  # (B, T, 2)
    if pos_weight is not None:
        # pos_weight is (2,) — apply per hand
        presence_loss = F.binary_cross_entropy_with_logits(
            presence_logits, target_presence, pos_weight=pos_weight, reduction="none"
        )
    else:
        presence_loss = F.binary_cross_entropy_with_logits(
            presence_logits, target_presence, reduction="none"
        )
    presence_loss = (presence_loss * mask).sum() / mask.sum()

    # Position loss (MSE only where note is present)
    position_pred = predictions["position_pred"]  # (B, T, 4)
    presence_mask = (target_presence > 0.5).float()  # (B, T, 2)
    # Expand presence mask for x and y per hand
    pos_mask = presence_mask.repeat_interleave(2, dim=-1)  # (B, T, 4)
    # Only compute loss for valid positions (not -1)
    valid_pos = (target_positions >= -0.99).float()
    pos_mask = pos_mask * valid_pos

    position_loss = F.mse_loss(position_pred, target_positions, reduction="none")
    position_loss = (position_loss * pos_mask).sum() / (pos_mask.sum() + 1e-8)

    # Rail loss (BCE)
    rail_logits = predictions["rail_logits"].squeeze(-1)  # (B, T)
    rail_loss = F.binary_cross_entropy_with_logits(
        rail_logits, target_rails, reduction="none"
    )
    rail_mask = mask.squeeze(-1)
    rail_loss = (rail_loss * rail_mask).sum() / rail_mask.sum()

    # Total
    total_loss = presence_loss + position_loss + 0.5 * rail_loss

    return total_loss, {
        "presence_loss": presence_loss.item(),
        "position_loss": position_loss.item(),
        "rail_loss": rail_loss.item(),
        "total_loss": total_loss.item(),
    }


def compute_metrics(
    predictions: dict,
    target_presence: torch.Tensor,
    target_positions: torch.Tensor,
    lengths: torch.Tensor,
    frame_ms: float = 20.0,
) -> dict:
    """
    Compute evaluation metrics.

    Returns:
        dict with note_recall, timing_error_ms, position_error
    """
    B, T = target_presence.shape[:2]
    device = target_presence.device

    mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)

    # Presence predictions
    presence_pred = (torch.sigmoid(predictions["presence_logits"]) > 0.5).float()
    presence_target = target_presence

    # Recall per hand
    recalls = []
    for hand in range(2):
        pred_hand = presence_pred[:, :, hand] * mask
        target_hand = presence_target[:, :, hand] * mask
        tp = (pred_hand * target_hand).sum()
        fn = ((1 - pred_hand) * target_hand).sum()
        recall = tp / (tp + fn + 1e-8)
        recalls.append(recall.item())

    avg_recall = sum(recalls) / len(recalls)

    # Precision per hand
    precisions = []
    for hand in range(2):
        pred_hand = presence_pred[:, :, hand] * mask
        target_hand = presence_target[:, :, hand] * mask
        tp = (pred_hand * target_hand).sum()
        fp = (pred_hand * (1 - target_hand)).sum()
        precision = tp / (tp + fp + 1e-8)
        precisions.append(precision.item())

    avg_precision = sum(precisions) / len(precisions)
    f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall + 1e-8)

    # Timing error: for predicted notes, how far is the nearest target note?
    # Simplified: compute mean absolute error on presence probability
    timing_error = torch.abs(
        torch.sigmoid(predictions["presence_logits"]) - target_presence
    )
    timing_error = (timing_error * mask.unsqueeze(-1)).sum() / (mask.sum() * 2 + 1e-8)
    timing_error_ms = timing_error.item() * frame_ms

    # Position error (only where both predict and target have notes)
    position_pred = predictions["position_pred"]
    pos_mask = (target_presence > 0.5).repeat_interleave(2, dim=-1)
    valid_pos = (target_positions >= -0.99).float()
    pos_mask = pos_mask * valid_pos

    pos_error = torch.abs(position_pred - target_positions)
    pos_error = (pos_error * pos_mask).sum() / (pos_mask.sum() + 1e-8)

    return {
        "note_recall": avg_recall,
        "precision": avg_precision,
        "f1": f1,
        "timing_error_ms": timing_error_ms,
        "position_error": pos_error.item(),
    }
