#!/usr/bin/env python3
"""
Evaluate generated beatmap quality against ground truth.

Metrics:
- Note recall: fraction of ground-truth notes matched
- Timing error: mean absolute time difference for matched notes
- Position error: Euclidean distance for matched notes
- False positive rate: predicted notes with no ground-truth match

Usage:
    python evaluate_map.py \
        --ground-truth dataset/parsed/0038583fdc1b90ba.json \
        --predicted /tmp/test_generated.synth \
        --difficulty Hard
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "projects", "synth-gen", "scripts"))
from synth_decryptor import read_synth


def load_ground_truth(parsed_path: str, difficulty: str = "Hard") -> list[dict]:
    with open(parsed_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    diff_data = data.get("difficulties", {}).get(difficulty)
    if not diff_data:
        return []
    return diff_data.get("notes", [])


def load_predicted(synth_path: str, password: str = "hC2*wE5R*qQzv@a!") -> list[dict]:
    data = read_synth(synth_path, password=password)
    return data.get("notes", [])


def match_notes(
    gt_notes: list[dict],
    pred_notes: list[dict],
    time_tolerance_ms: float = 100.0,
    hand_must_match: bool = True,
) -> tuple[list, list, list]:
    """
    Match predicted notes to ground truth.

    Returns:
        matches: list of (gt_note, pred_note, time_diff, pos_diff)
        unmatched_gt: list of ground truth notes with no match
        unmatched_pred: list of predicted notes with no match
    """
    gt_sorted = sorted(gt_notes, key=lambda n: n["time"])
    pred_sorted = sorted(pred_notes, key=lambda n: n["time"])

    matches = []
    unmatched_gt = []
    matched_pred_indices = set()

    for gt in gt_sorted:
        best_match = None
        best_diff = float("inf")

        for i, pred in enumerate(pred_sorted):
            if i in matched_pred_indices:
                continue

            time_diff = abs(gt["time"] - pred["time"])
            if time_diff > time_tolerance_ms:
                continue

            if hand_must_match and gt.get("type") != pred.get("type"):
                continue

            # Compute position distance
            gx, gy = gt.get("x", 0), gt.get("y", 0)
            px, py = pred.get("x", 0), pred.get("y", 0)
            pos_diff = ((gx - px) ** 2 + (gy - py) ** 2) ** 0.5

            # Combined score (time is more important)
            score = time_diff + pos_diff * 1000  # weight position heavily
            if score < best_diff:
                best_diff = score
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


def evaluate(
    gt_notes: list[dict],
    pred_notes: list[dict],
    time_tolerance_ms: float = 100.0,
) -> dict:
    matches, unmatched_gt, unmatched_pred = match_notes(gt_notes, pred_notes, time_tolerance_ms)

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
        "timing_error_ms": np.mean(timing_errors) if timing_errors else 0.0,
        "timing_error_std_ms": np.std(timing_errors) if timing_errors else 0.0,
        "position_error": np.mean(pos_errors) if pos_errors else 0.0,
        "position_error_std": np.std(pos_errors) if pos_errors else 0.0,
        "unmatched_gt": len(unmatched_gt),
        "unmatched_pred": len(unmatched_pred),
        "false_positive_rate": len(unmatched_pred) / pred_count if pred_count > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated beatmap against ground truth")
    parser.add_argument("--ground-truth", required=True, help="Path to parsed JSON ground truth")
    parser.add_argument("--predicted", required=True, help="Path to generated .synth file")
    parser.add_argument("--difficulty", default="Hard")
    parser.add_argument("--time-tolerance", type=float, default=100.0, help="Time matching tolerance in ms")
    args = parser.parse_args()

    gt_notes = load_ground_truth(args.ground_truth, args.difficulty)
    pred_notes = load_predicted(args.predicted)

    print(f"[+] Ground truth: {len(gt_notes)} notes")
    print(f"[+] Predicted: {len(pred_notes)} notes")
    print()

    results = evaluate(gt_notes, pred_notes, args.time_tolerance)

    print("=" * 50)
    print("  Evaluation Results")
    print("=" * 50)
    print(f"  Ground truth notes     : {results['gt_count']}")
    print(f"  Predicted notes        : {results['pred_count']}")
    print(f"  Matched notes          : {results['matches']}")
    print(f"  Unmatched GT           : {results['unmatched_gt']}")
    print(f"  Unmatched Pred (FP)    : {results['unmatched_pred']}")
    print("-" * 50)
    print(f"  Recall                 : {results['recall']:.3f}")
    print(f"  Precision              : {results['precision']:.3f}")
    print(f"  F1 Score               : {results['f1']:.3f}")
    print(f"  Timing Error (mean)    : {results['timing_error_ms']:.1f} ms")
    print(f"  Timing Error (std)     : {results['timing_error_std_ms']:.1f} ms")
    print(f"  Position Error (mean)  : {results['position_error']:.4f}")
    print(f"  Position Error (std)   : {results['position_error_std']:.4f}")
    print(f"  False Positive Rate    : {results['false_positive_rate']:.3f}")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
