#!/usr/bin/env python3
"""
Visualize beatmap predictions vs ground truth.

Usage:
    python visualize_predictions.py --checkpoint models/checkpoints/best_model.pt --features dataset/features --index 0
"""

import argparse
import json
import os

import numpy as np
import torch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.baseline import BaselineBeatmapModel


def visualize_predictions(
    checkpoint_path: str,
    features_dir: str,
    output_path: str,
    index: int = 0,
) -> None:
    """Load model, run inference on one sample, save visualization."""
    device = torch.device("cpu")

    # Load model
    model = BaselineBeatmapModel()
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # Load feature file
    files = sorted([f for f in os.listdir(features_dir) if f.endswith(".npz")])
    feat_path = os.path.join(features_dir, files[index])
    data = np.load(feat_path)

    audio_feat = torch.from_numpy(np.random.randn(1, data["note_presence"].shape[0], 80)).float()

    with torch.no_grad():
        preds = model(audio_feat)

    # Extract predictions
    presence_pred = torch.sigmoid(preds["presence_logits"]).squeeze(0).numpy()
    position_pred = preds["position_pred"].squeeze(0).numpy()

    presence_target = data["note_presence"]
    position_target = data["note_positions"]

    # Create simple text report
    report = []
    report.append("# Beatmap Prediction Visualization\n")
    report.append(f"File: {files[index]}\n")
    report.append(f"Frames: {presence_target.shape[0]}\n\n")

    report.append("## Note Presence (Ground Truth vs Predicted)\n")
    report.append("| Frame | Right GT | Right Pred | Left GT | Left Pred |\n")
    report.append("|-------|----------|------------|---------|-----------|\n")

    # Sample every 50 frames
    for t in range(0, presence_target.shape[0], 50):
        r_gt = presence_target[t, 0]
        r_pred = presence_pred[t, 0]
        l_gt = presence_target[t, 1]
        l_pred = presence_pred[t, 1]
        report.append(f"| {t:5d} | {r_gt:.2f} | {r_pred:.2f} | {l_gt:.2f} | {l_pred:.2f} |\n")

    report.append("\n## Sample Positions (Ground Truth vs Predicted)\n")
    report.append("| Frame | Hand | GT X | Pred X | GT Y | Pred Y |\n")
    report.append("|-------|------|------|--------|------|--------|\n")

    for t in range(0, min(presence_target.shape[0], 500), 25):
        for hand, name in [(0, "Right"), (1, "Left")]:
            if presence_target[t, hand] > 0.5:
                gt_x = position_target[t, hand, 0]
                gt_y = position_target[t, hand, 1]
                pred_x = position_pred[t, hand * 2]
                pred_y = position_pred[t, hand * 2 + 1]
                report.append(f"| {t:5d} | {name:5s} | {gt_x:+.3f} | {pred_x:+.3f} | {gt_y:+.3f} | {pred_y:+.3f} |\n")

    with open(output_path, "w") as f:
        f.writelines(report)

    print(f"Saved visualization to {output_path}")


def plot_loss_curves(history_path: str, output_path: str) -> None:
    """Generate ASCII loss curves from training history."""
    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train"]["loss"] for h in history]
    val_loss = [h["val"]["loss"] for h in history]
    val_recall = [h["val"]["note_recall"] for h in history]

    report = []
    report.append("# Training Loss Curves\n\n")

    report.append("## Total Loss\n\n")
    report.append("```\n")
    report.append(plot_ascii_curve(epochs, train_loss, "Train Loss", val_loss, "Val Loss"))
    report.append("```\n\n")

    report.append("## Note Recall\n\n")
    report.append("```\n")
    report.append(plot_ascii_curve(epochs, val_recall, "Recall", None, None))
    report.append("```\n\n")

    with open(output_path, "w") as f:
        f.writelines(report)

    print(f"Saved loss curves to {output_path}")


def plot_ascii_curve(x, y1, label1, y2=None, label2=None, width=60, height=15) -> str:
    """Plot a simple ASCII line chart."""
    if not y1:
        return "No data\n"

    min_y = min(min(y1), min(y2)) if y2 else min(y1)
    max_y = max(max(y1), max(y2)) if y2 else max(y1)
    range_y = max_y - min_y if max_y != min_y else 1.0

    lines = []
    lines.append(f"{label1} (max={max_y:.4f}, min={min_y:.4f})")
    if y2 and label2:
        lines.append(f"{label2}")

    for row in range(height):
        y_val = max_y - (row / (height - 1)) * range_y
        row_str = f"{y_val:.4f} |"
        for i in range(len(y1)):
            col = int(i / len(y1) * width)
            # Simple plotting - just mark if close
            y1_norm = (y1[i] - min_y) / range_y
            y1_row = int(y1_norm * (height - 1))
            if y2:
                y2_norm = (y2[i] - min_y) / range_y
                y2_row = int(y2_norm * (height - 1))
                if y1_row == row and y2_row == row:
                    row_str += "*"
                elif y1_row == row:
                    row_str += "."
                elif y2_row == row:
                    row_str += "o"
                else:
                    row_str += " "
            else:
                if y1_row == row:
                    row_str += "."
                else:
                    row_str += " "
        lines.append(row_str)

    lines.append("       +" + "-" * width)
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Visualize beatmap predictions")
    parser.add_argument("--checkpoint", default="models/checkpoints/best_model.pt")
    parser.add_argument("--features-dir", default="dataset/features")
    parser.add_argument("--history", default="models/checkpoints/history.json")
    parser.add_argument("--output", default="models/checkpoints/visualization.md")
    parser.add_argument("--loss-output", default="models/checkpoints/loss_curves.md")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    if os.path.exists(args.checkpoint):
        visualize_predictions(args.checkpoint, args.features_dir, args.output, args.index)
    else:
        print(f"Checkpoint not found: {args.checkpoint}")

    if os.path.exists(args.history):
        plot_loss_curves(args.history, args.loss_output)
    else:
        print(f"History not found: {args.history}")


if __name__ == "__main__":
    main()
