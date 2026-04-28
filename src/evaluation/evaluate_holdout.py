#!/usr/bin/env python3
"""
Holdout-set evaluation for the baseline beatmap model.

Evaluates a trained model on held-out maps (never seen during training) to measure
generalization. Computes note-level metrics via temporal matching, not just frame-level
presence accuracy.

Usage:
    python evaluate_holdout.py \
        --checkpoint models/checkpoints/best_model.pt \
        --features-dir dataset/features \
        --audio-features-dir dataset/audio_features \
        --output-dir evaluation/holdout \
        --num-maps 100 \
        --difficulty Hard \
        --train-split 0.8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional  # Added for compatibility

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.features.feature_engineering import SynthBeatmapDataset
from src.models.baseline import BaselineBeatmapModel


def collate_fn(batch):
    """Collate variable-length sequences (same as train_baseline.py)."""
    audio_feats, occ, pos, pres, lengths = zip(*batch)
    audio_feats = torch.stack(audio_feats)
    occ = torch.stack(occ)
    pos = torch.stack(pos)
    pres = torch.stack(pres)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return audio_feats, occ, pos, pres, lengths


def extract_predicted_notes(
    presence_logits: torch.Tensor,
    position_pred: torch.Tensor,
    frame_ms: float = 20.0,
    presence_threshold: float = 0.5,
    min_gap_ms: float = 40.0,
) -> list[dict]:
    """
    Convert model frame-level outputs into discrete note events.

    Args:
        presence_logits: (T, 2) logits per hand
        position_pred:   (T, 4) continuous [right_x, right_y, left_x, left_y]
        frame_ms:        duration of each frame in ms
        presence_threshold: sigmoid threshold for "note exists"
        min_gap_ms:      minimum time gap between consecutive notes (dedup)

    Returns:
        List of note dicts: {"time": float_ms, "x": float, "y": float, "type": int}
    """
    presence = torch.sigmoid(presence_logits).cpu().numpy()  # (T, 2)
    positions = position_pred.cpu().numpy()                    # (T, 4)

    notes = []
    min_gap_frames = int(min_gap_ms / frame_ms)

    for hand in range(2):
        pos_offset = hand * 2
        active = presence[:, hand] > presence_threshold

        # Find contiguous regions of active frames
        frame_indices = np.where(active)[0]
        if len(frame_indices) == 0:
            continue

        # Group consecutive frames
        groups = []
        current_group = [frame_indices[0]]
        for idx in frame_indices[1:]:
            if idx == current_group[-1] + 1:
                current_group.append(idx)
            else:
                groups.append(current_group)
                current_group = [idx]
        groups.append(current_group)

        last_note_frame = -min_gap_frames - 1

        for group in groups:
            # Pick the frame with highest presence within the group
            best_idx = group[np.argmax(presence[group, hand])]

            # Enforce minimum gap
            if best_idx - last_note_frame < min_gap_frames:
                continue
            last_note_frame = best_idx

            time_ms = float(best_idx * frame_ms)
            x = float(positions[best_idx, pos_offset])
            y = float(positions[best_idx, pos_offset + 1])

            notes.append({
                "time": time_ms,
                "x": x,
                "y": y,
                "type": hand,  # 0=right, 1=left
            })

    # Sort by time
    notes.sort(key=lambda n: n["time"])
    return notes


def match_notes(
    gt_notes: list[dict],
    pred_notes: list[dict],
    time_tolerance_ms: float = 100.0,
    hand_must_match: bool = True,
) -> tuple[list, list, list]:
    """
    Match predicted notes to ground truth.

    Returns:
        matches:      list of (gt, pred, time_diff, pos_diff)
        unmatched_gt: ground truth notes with no match
        unmatched_pred: predicted notes with no match (false positives)
    """
    gt_sorted = sorted(gt_notes, key=lambda n: n["time"])
    pred_sorted = sorted(pred_notes, key=lambda n: n["time"])

    matches = []
    unmatched_gt = []
    matched_pred_indices = set()

    for gt in gt_sorted:
        best_match = None
        best_score = float("inf")

        for i, pred in enumerate(pred_sorted):
            if i in matched_pred_indices:
                continue

            time_diff = abs(gt["time"] - pred["time"])
            if time_diff > time_tolerance_ms:
                continue

            if hand_must_match and gt.get("type") != pred.get("type"):
                continue

            # Combined score: time prioritized, position secondary
            gx, gy = gt.get("x", 0), gt.get("y", 0)
            px, py = pred.get("x", 0), pred.get("y", 0)
            pos_diff = ((gx - px) ** 2 + (gy - py) ** 2) ** 0.5
            score = time_diff + pos_diff * 1000  # position in ~units, scale to ms

            if score < best_score:
                best_score = score
                best_match = i

        if best_match is not None:
            pred = pred_sorted[best_match]
            time_diff = abs(gt["time"] - pred["time"])
            gx, gy = gt.get("x", 0), gt.get("y", 0)
            px, py = pred.get("x", 0), pred.get("y", 0)
            pos_diff = ((gx - px) ** 2 + (gy - py) ** 2) ** 0.5
            matches.append((gt, pred, time_diff, pos_diff))
            matched_pred_indices.add(best_match)
        else:
            unmatched_gt.append(gt)

    unmatched_pred = [pred_sorted[i] for i in range(len(pred_sorted)) if i not in matched_pred_indices]
    return matches, unmatched_gt, unmatched_pred


def evaluate_map(
    model: nn.Module,
    audio_feats: torch.Tensor,
    gt_notes: list[dict],
    device: torch.device,
    time_tolerance_ms: float = 100.0,
    presence_threshold: float = 0.5,
) -> dict:
    """
    Evaluate a single map: run inference, extract notes, match to ground truth.

    Returns metric dict.
    """
    model.eval()
    with torch.no_grad():
        audio_feats = audio_feats.unsqueeze(0).to(device)  # (1, T, F)
        predictions = model(audio_feats)
        presence_logits = predictions["presence_logits"].squeeze(0)  # (T, 2)
        position_pred = predictions["position_pred"].squeeze(0)       # (T, 4)

    pred_notes = extract_predicted_notes(
        presence_logits, position_pred,
        presence_threshold=presence_threshold,
    )

    matches, unmatched_gt, unmatched_pred = match_notes(
        gt_notes, pred_notes, time_tolerance_ms=time_tolerance_ms,
    )

    gt_count = len(gt_notes)
    pred_count = len(pred_notes)
    match_count = len(matches)

    recall = match_count / gt_count if gt_count > 0 else 0.0
    precision = match_count / pred_count if pred_count > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    timing_errors = [m[2] for m in matches]
    pos_errors = [m[3] for m in matches]

    return {
        "gt_count": gt_count,
        "pred_count": pred_count,
        "matches": match_count,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "timing_error_ms": float(np.mean(timing_errors)) if timing_errors else 0.0,
        "timing_error_std_ms": float(np.std(timing_errors)) if timing_errors else 0.0,
        "position_error": float(np.mean(pos_errors)) if pos_errors else 0.0,
        "position_error_std": float(np.std(pos_errors)) if pos_errors else 0.0,
        "unmatched_gt": len(unmatched_gt),
        "unmatched_pred": len(unmatched_pred),
        "false_positive_rate": len(unmatched_pred) / pred_count if pred_count > 0 else 0.0,
        "pred_notes": pred_notes,  # include for debug
    }


def load_ground_truth_notes(parsed_path: str, difficulty: str) -> list[dict]:
    """Load notes from parsed JSON."""
    with open(parsed_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    diff_data = data.get("difficulties", {}).get(difficulty)
    if not diff_data:
        return []
    return diff_data.get("notes", [])


def run_holdout_evaluation(
    checkpoint_path: str,
    features_dir: str,
    audio_features_dir: str | None,
    parsed_dir: str,
    output_dir: str,
    num_maps: int,
    difficulty: str,
    train_split: float,
    batch_size: int,
    device_str: str,
    time_tolerance_ms: float,
    presence_threshold: float,
) -> dict:
    """Main evaluation routine."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device_str if torch.backends.mps.is_available() else "cpu")
    print(f"[+] Device: {device}")

    # Load dataset
    dataset = SynthBeatmapDataset(
        features_dir,
        difficulty=difficulty,
        audio_features_dir=audio_features_dir,
    )
    print(f"[+] Dataset: {len(dataset)} maps")

    # Deterministic split
    num_total = len(dataset)
    num_train = int(num_total * train_split)
    indices = list(range(num_total))
    # Use a fixed seed for reproducibility
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    train_indices = indices[:num_train]
    holdout_indices = indices[num_train:]

    # Subset holdout to requested size
    if num_maps > 0:
        holdout_indices = holdout_indices[:num_maps]

    print(f"[+] Train: {len(train_indices)} | Holdout: {len(holdout_indices)}")

    # Load model
    model = BaselineBeatmapModel().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[+] Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Build holdout DataLoader
    holdout_subset = Subset(dataset, holdout_indices)
    holdout_loader = DataLoader(
        holdout_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Pre-build UUID -> parsed path mapping
    parsed_files = {p.stem: p for p in Path(parsed_dir).glob("*.json")}

    # Evaluation loop
    results_per_map = []
    totals = {
        "gt_count": 0, "pred_count": 0, "matches": 0,
        "unmatched_gt": 0, "unmatched_pred": 0,
    }
    all_timing_errors = []
    all_pos_errors = []

    print(f"[+] Evaluating {len(holdout_subset)} holdout maps...")
    for batch_idx, batch in enumerate(holdout_loader):
        audio_feats, occ, pos, pres, lengths = [b.to(device) for b in batch]
        B = audio_feats.shape[0]

        for i in range(B):
            # Get the UUID for this sample
            global_idx = holdout_indices[batch_idx * batch_size + i]
            uuid = dataset.valid_files[global_idx].stem

            # Load ground truth notes
            parsed_path = parsed_files.get(uuid)
            if not parsed_path or not parsed_path.exists():
                print(f"  [!] Parsed JSON not found for {uuid}, skipping")
                continue

            gt_notes = load_ground_truth_notes(str(parsed_path), difficulty)
            if not gt_notes:
                continue

            result = evaluate_map(
                model,
                audio_feats[i, :lengths[i]],
                gt_notes,
                device,
                time_tolerance_ms=time_tolerance_ms,
                presence_threshold=presence_threshold,
            )
            result["uuid"] = uuid
            results_per_map.append(result)

            totals["gt_count"] += result["gt_count"]
            totals["pred_count"] += result["pred_count"]
            totals["matches"] += result["matches"]
            totals["unmatched_gt"] += result["unmatched_gt"]
            totals["unmatched_pred"] += result["unmatched_pred"]
            if result["timing_error_ms"] > 0:
                all_timing_errors.append(result["timing_error_ms"])
            if result["position_error"] > 0:
                all_pos_errors.append(result["position_error"])

    # Aggregate metrics
    if totals["gt_count"] > 0:
        overall_recall = totals["matches"] / totals["gt_count"]
        overall_precision = totals["matches"] / totals["pred_count"] if totals["pred_count"] > 0 else 0.0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    else:
        overall_recall = overall_precision = overall_f1 = 0.0

    summary = {
        "checkpoint": checkpoint_path,
        "num_holdout_maps": len(results_per_map),
        "difficulty": difficulty,
        "time_tolerance_ms": time_tolerance_ms,
        "presence_threshold": presence_threshold,
        "overall": {
            "gt_count": totals["gt_count"],
            "pred_count": totals["pred_count"],
            "matches": totals["matches"],
            "recall": overall_recall,
            "precision": overall_precision,
            "f1": overall_f1,
            "timing_error_ms": float(np.mean(all_timing_errors)) if all_timing_errors else 0.0,
            "timing_error_std_ms": float(np.std(all_timing_errors)) if len(all_timing_errors) > 1 else 0.0,
            "position_error": float(np.mean(all_pos_errors)) if all_pos_errors else 0.0,
            "position_error_std": float(np.std(all_pos_errors)) if len(all_pos_errors) > 1 else 0.0,
            "false_positive_rate": totals["unmatched_pred"] / totals["pred_count"] if totals["pred_count"] > 0 else 0.0,
        },
        "per_map": [
            {
                "uuid": r["uuid"],
                "gt_count": r["gt_count"],
                "pred_count": r["pred_count"],
                "recall": r["recall"],
                "precision": r["precision"],
                "f1": r["f1"],
                "timing_error_ms": r["timing_error_ms"],
                "position_error": r["position_error"],
                "false_positive_rate": r["false_positive_rate"],
            }
            for r in results_per_map
        ],
    }

    # Save results
    out_path = os.path.join(output_dir, f"holdout_results_{difficulty.lower()}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[+] Saved results to {out_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("  HOLDOUT EVALUATION SUMMARY")
    print("=" * 60)
    o = summary["overall"]
    print(f"  Maps evaluated         : {summary['num_holdout_maps']}")
    print(f"  Total GT notes         : {o['gt_count']}")
    print(f"  Total predicted notes  : {o['pred_count']}")
    print(f"  Matched notes          : {o['matches']}")
    print("-" * 60)
    print(f"  Recall                 : {o['recall']:.3f}")
    print(f"  Precision              : {o['precision']:.3f}")
    print(f"  F1 Score               : {o['f1']:.3f}")
    print(f"  Timing Error (mean)    : {o['timing_error_ms']:.1f} ms")
    print(f"  Timing Error (std)     : {o['timing_error_std_ms']:.1f} ms")
    print(f"  Position Error (mean)  : {o['position_error']:.4f}")
    print(f"  Position Error (std)   : {o['position_error_std']:.4f}")
    print(f"  False Positive Rate    : {o['false_positive_rate']:.3f}")
    print("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline model on holdout set")
    parser.add_argument("--checkpoint", default="models/checkpoints/best_model.pt")
    parser.add_argument("--features-dir", default="/Volumes/Second-Brain-1/AI/Synth/dataset/features")
    parser.add_argument("--parsed-dir", default="/Volumes/Second-Brain-1/AI/Synth/dataset/parsed")
    parser.add_argument("--audio-features-dir", default="/Volumes/Second-Brain-1/AI/Synth/dataset/audio_features")
    parser.add_argument("--output-dir", default="/Volumes/Second-Brain-1/AI/Synth/evaluation/holdout")
    parser.add_argument("--num-maps", type=int, default=100, help="Number of holdout maps to evaluate (0 = all)")
    parser.add_argument("--difficulty", default="Hard")
    parser.add_argument("--train-split", type=float, default=0.8, help="Fraction of data used for training (rest = holdout)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--time-tolerance", type=float, default=100.0)
    parser.add_argument("--presence-threshold", type=float, default=0.5)
    args = parser.parse_args()

    run_holdout_evaluation(
        checkpoint_path=args.checkpoint,
        features_dir=args.features_dir,
        audio_features_dir=args.audio_features_dir,
        parsed_dir=args.parsed_dir,
        output_dir=args.output_dir,
        num_maps=args.num_maps,
        difficulty=args.difficulty,
        train_split=args.train_split,
        batch_size=args.batch_size,
        device_str=args.device,
        time_tolerance_ms=args.time_tolerance,
        presence_threshold=args.presence_threshold,
    )


if __name__ == "__main__":
    sys.exit(main())
