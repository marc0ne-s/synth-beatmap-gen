"""
Focal loss for extreme class imbalance in beatmap presence prediction.

Down-weights easy negatives so the model focuses on hard examples and positives.
"""

import torch
import torch.nn.functional as F


def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Focal loss for binary classification.

    Args:
        logits: (B, T, 2) raw logits per hand
        targets: (B, T, 2) binary targets
        alpha: weighting factor for positive class (default 0.25)
        gamma: focusing parameter (default 2.0)
        reduction: "mean", "sum", or "none"

    Returns:
        Scalar loss if reduction="mean", else tensor of same shape as inputs
    """
    # BCE with logits
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    # p_t = p if y=1, else 1-p
    prob = torch.sigmoid(logits)
    p_t = prob * targets + (1 - prob) * (1 - targets)

    # focal weight: (1 - p_t)^gamma
    focal_weight = (1 - p_t) ** gamma

    # alpha weighting
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

    loss = alpha_t * focal_weight * bce

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def compute_focal_loss(
    predictions: dict,
    target_presence: torch.Tensor,
    target_positions: torch.Tensor,
    target_rails: torch.Tensor,
    lengths: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> tuple[torch.Tensor, dict]:
    """
    Multi-task loss with focal loss for presence.
    """
    B, T = target_presence.shape[:2]

    mask = torch.arange(T, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1)

    # Presence loss: focal instead of weighted BCE
    presence_logits = predictions["presence_logits"]
    presence_loss_raw = focal_loss_with_logits(
        presence_logits, target_presence, alpha=alpha, gamma=gamma, reduction="none"
    )
    presence_loss = (presence_loss_raw * mask).sum() / mask.sum()

    # Position loss (MSE only where note is present)
    position_pred = predictions["position_pred"]
    presence_mask = (target_presence > 0.5).float()
    pos_mask = presence_mask.repeat_interleave(2, dim=-1)
    valid_pos = (target_positions >= -0.99).float()
    pos_mask = pos_mask * valid_pos

    position_loss = F.mse_loss(position_pred, target_positions, reduction="none")
    position_loss = (position_loss * pos_mask).sum() / (pos_mask.sum() + 1e-8)

    # Rail loss (BCE)
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
