"""
Upgraded beatmap prediction model for SynthRiders.

Larger capacity version of BaselineBeatmapModel:
    - Encoder: 256 hidden channels, 4 layers
    - Decoder: 512 hidden, 4-layer LSTM
    - Optional difficulty conditioning via embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class Conv1DEncoder(nn.Module):
    """Encode audio features into a latent sequence."""

    def __init__(
        self,
        in_channels: int = 80,
        hidden_channels: int = 256,
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
        x = x.transpose(1, 2)  # (B, F, T)
        x = self.encoder(x)
        x = x.transpose(1, 2)  # (B, T, D)
        return x


class BeatmapDecoder(nn.Module):
    """Decode latent sequence into per-frame beatmap predictions."""

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_difficulties: int = 5,
        use_difficulty_embedding: bool = False,
    ):
        super().__init__()
        self.use_difficulty_embedding = use_difficulty_embedding

        if use_difficulty_embedding:
            self.difficulty_embed = nn.Embedding(num_difficulties, hidden_dim)
            lstm_input = latent_dim
        else:
            lstm_input = latent_dim

        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )

        # Layer norm on LSTM output for stability
        self.output_norm = nn.LayerNorm(hidden_dim)

        # Multi-task heads
        self.presence_head = nn.Linear(hidden_dim, 2)
        self.position_head = nn.Linear(hidden_dim, 4)
        self.rail_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, difficulty: Optional[torch.Tensor] = None) -> dict:
        out, _ = self.lstm(x)
        out = self.output_norm(out)

        if self.use_difficulty_embedding and difficulty is not None:
            diff_emb = self.difficulty_embed(difficulty).unsqueeze(1)  # (B, 1, D)
            out = out + diff_emb  # broadcast across time

        presence_logits = self.presence_head(out)
        position_pred = self.position_head(out)
        rail_logits = self.rail_head(out)

        return {
            "presence_logits": presence_logits,
            "position_pred": position_pred,
            "rail_logits": rail_logits,
        }


class LargeBeatmapModel(nn.Module):
    """End-to-end beatmap prediction model with increased capacity."""

    def __init__(
        self,
        audio_features: int = 80,
        encoder_hidden: int = 256,
        encoder_layers: int = 4,
        decoder_hidden: int = 512,
        decoder_layers: int = 4,
        use_difficulty_embedding: bool = False,
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
            use_difficulty_embedding=use_difficulty_embedding,
        )

    def forward(self, audio_features: torch.Tensor, difficulty: Optional[torch.Tensor] = None) -> dict:
        latent = self.encoder(audio_features)
        return self.decoder(latent, difficulty)


def compute_loss(
    predictions: dict,
    target_presence: torch.Tensor,
    target_positions: torch.Tensor,
    target_rails: torch.Tensor,
    lengths: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict]:
    """Compute multi-task loss (same as baseline)."""
    B, T = target_presence.shape[:2]

    mask = torch.arange(T, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1)

    presence_logits = predictions["presence_logits"]
    if pos_weight is not None:
        presence_loss = F.binary_cross_entropy_with_logits(
            presence_logits, target_presence, pos_weight=pos_weight, reduction="none"
        )
    else:
        presence_loss = F.binary_cross_entropy_with_logits(
            presence_logits, target_presence, reduction="none"
        )
    presence_loss = (presence_loss * mask).sum() / mask.sum()

    position_pred = predictions["position_pred"]
    presence_mask = (target_presence > 0.5).float()
    pos_mask = presence_mask.repeat_interleave(2, dim=-1)
    valid_pos = (target_positions >= -0.99).float()
    pos_mask = pos_mask * valid_pos

    position_loss = F.mse_loss(position_pred, target_positions, reduction="none")
    position_loss = (position_loss * pos_mask).sum() / (pos_mask.sum() + 1e-8)

    rail_logits = predictions["rail_logits"].squeeze(-1)
    rail_loss = F.binary_cross_entropy_with_logits(
        rail_logits, target_rails, reduction="none"
    )
    rail_mask = mask.squeeze(-1)
    rail_loss = (rail_loss * rail_mask).sum() / rail_mask.sum()

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
    """Compute evaluation metrics with precision and F1."""
    B, T = target_presence.shape[:2]
    device = target_presence.device

    mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)

    presence_pred = (torch.sigmoid(predictions["presence_logits"]) > 0.5).float()
    presence_target = target_presence

    recalls = []
    precisions = []
    for hand in range(2):
        pred_hand = presence_pred[:, :, hand] * mask
        target_hand = presence_target[:, :, hand] * mask
        tp = (pred_hand * target_hand).sum()
        fn = ((1 - pred_hand) * target_hand).sum()
        fp = (pred_hand * (1 - target_hand)).sum()
        recall = tp / (tp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recalls.append(recall.item())
        precisions.append(precision.item())

    avg_recall = sum(recalls) / len(recalls)
    avg_precision = sum(precisions) / len(precisions)
    f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall + 1e-8)

    timing_error = torch.abs(
        torch.sigmoid(predictions["presence_logits"]) - target_presence
    )
    timing_error = (timing_error * mask.unsqueeze(-1)).sum() / (mask.sum() * 2 + 1e-8)
    timing_error_ms = timing_error.item() * frame_ms

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
