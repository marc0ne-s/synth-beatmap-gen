#!/usr/bin/env python3
"""
batch_feature_generation.py — The Extraction Engine

Batch-processes the full SynthRiders corpus:
  1. Beatmap features (parsed JSON → NPZ tensors) for ALL difficulties
  2. Audio mel spectrograms (OGG → NPZ) for all maps with audio

Usage:
    # Pilot batch (100 maps)
    python scripts/batch_feature_generation.py --batch-size 100

    # Full run in 500-map batches
    python scripts/batch_feature_generation.py --batch-size 500

    # Resume from where we left off (reads manifest_progress.json)
    python scripts/batch_feature_generation.py --resume --batch-size 500

    # Features only (skip audio)
    python scripts/batch_feature_generation.py --batch-size 500 --skip-audio

    # Audio only (skip features)
    python scripts/batch_feature_generation.py --batch-size 500 --skip-features
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np

# Project imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.feature_engineering import (
    extract_beatmap_features,
    save_features,
)

# Audio constants (matching extract_audio_batch.py)
SR = 22050
N_MELS = 80
HOP_LENGTH = 512
FRAME_MS = 20.0

# All difficulty tiers to extract
ALL_DIFFICULTIES = ["Easy", "Normal", "Hard", "Expert", "Master", "Custom"]

# Paths
PARSED_DIR = ROOT / "dataset" / "parsed"
FEATURES_DIR = ROOT / "dataset" / "features"
AUDIO_DIR = ROOT / "dataset" / "audio_features"
EXTRACTED_DIR = ROOT / "dataset" / "extracted"
MANIFEST_PATH = ROOT / "dataset" / "manifest_progress.json"


def load_manifest() -> dict:
    """Load or create the progress manifest."""
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r") as f:
            return json.load(f)
    return {
        "total_maps": 0,
        "features_done": [],
        "audio_done": [],
        "features_failed": [],
        "audio_failed": [],
        "last_updated": None,
    }


def save_manifest(manifest: dict) -> None:
    """Save progress manifest atomically."""
    manifest["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp = str(MANIFEST_PATH) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, str(MANIFEST_PATH))


def get_all_uuids() -> list[str]:
    """Get all UUID stems from the parsed directory."""
    uuids = []
    for f in sorted(PARSED_DIR.iterdir()):
        if f.suffix == ".json" and f.stem != "index":
            uuids.append(f.stem)
    return uuids


def extract_features_for_map(uuid: str) -> dict:
    """
    Extract beatmap features for a single map across ALL available difficulties.

    Returns dict with status info for the manifest.
    """
    parsed_path = PARSED_DIR / f"{uuid}.json"
    result = {
        "uuid": uuid,
        "status": "success",
        "difficulties": [],
        "errors": [],
    }

    try:
        with open(parsed_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        result["status"] = "failed"
        result["errors"].append(f"JSON read: {str(e)[:80]}")
        return result

    if data.get("status") != "success":
        result["status"] = "skipped"
        result["errors"].append("parsed status != success")
        return result

    available_diffs = list(data.get("difficulties", {}).keys())
    if not available_diffs:
        result["status"] = "skipped"
        result["errors"].append("no difficulties")
        return result

    any_success = False
    for diff in available_diffs:
        # Output path: uuid_difficulty.npz (e.g. abc123_Hard.npz)
        out_path = FEATURES_DIR / f"{uuid}_{diff}.npz"

        # Legacy path without difficulty suffix (Hard only, from Phase 1)
        legacy_path = FEATURES_DIR / f"{uuid}.npz"

        # Skip if already exists
        if out_path.exists():
            result["difficulties"].append(diff)
            any_success = True
            continue

        # Check if legacy file exists for Hard difficulty
        if diff == "Hard" and legacy_path.exists():
            result["difficulties"].append(diff)
            any_success = True
            continue

        try:
            features = extract_beatmap_features(str(parsed_path), difficulty=diff)
            if features is not None:
                save_features(features, str(out_path))
                result["difficulties"].append(diff)
                any_success = True
            # features is None = difficulty has 0 notes, not an error
        except Exception as e:
            result["errors"].append(f"{diff}: {str(e)[:60]}")

    if not any_success and not result["errors"]:
        result["status"] = "skipped"
    elif not any_success:
        result["status"] = "failed"

    return result


def extract_audio_for_map(uuid: str) -> dict:
    """
    Extract mel spectrogram from OGG audio for a single map.

    Returns dict with status info.
    """
    result = {
        "uuid": uuid,
        "status": "success",
        "error": None,
    }

    out_path = AUDIO_DIR / f"{uuid}.npz"

    # Skip if already exists
    if out_path.exists():
        result["status"] = "exists"
        return result

    # Find OGG file
    ogg_path = EXTRACTED_DIR / uuid / "audio.ogg"
    if not ogg_path.exists():
        result["status"] = "no_audio"
        result["error"] = "audio.ogg not found"
        return result

    # Get target frames from parsed JSON
    parsed_path = PARSED_DIR / f"{uuid}.json"
    try:
        with open(parsed_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        result["status"] = "failed"
        result["error"] = f"JSON read: {str(e)[:60]}"
        return result

    # Find the first non-empty difficulty to get duration
    target_frames = None
    for diff in ALL_DIFFICULTIES:
        diff_data = data.get("difficulties", {}).get(diff)
        if diff_data and diff_data.get("note_count", 0) > 0:
            notes = diff_data["notes"]
            max_time_ms = max(float(n["time"]) for n in notes)
            duration_ms = min(max_time_ms + 5000, 300_000)
            target_frames = min(int(duration_ms / FRAME_MS) + 1, 15_000)
            break

    if target_frames is None:
        result["status"] = "failed"
        result["error"] = "no valid difficulty for frame count"
        return result

    # Extract mel spectrogram
    try:
        import librosa
        import torch
        import torch.nn.functional as F

        y, sr = librosa.load(str(ogg_path), sr=SR, mono=True)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_t = torch.from_numpy(mel_db).unsqueeze(0)  # (1, n_mels, T_audio)
        mel_t = F.interpolate(mel_t, size=target_frames, mode="linear", align_corners=False)
        mel_np = mel_t.squeeze(0).permute(1, 0).numpy().astype(np.float32)

        bpm = data.get("metadata", {}).get("bpm", 120.0)
        np.savez_compressed(str(out_path), audio_mel=mel_np, uuid=uuid, bpm=bpm)

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)[:80]

    return result


def run_feature_batch(uuids: list[str], manifest: dict, workers: int = 1) -> dict:
    """Run feature extraction on a batch of UUIDs."""
    stats = {"success": 0, "skipped": 0, "failed": 0, "exists": 0, "total_diffs": 0}
    t0 = time.time()

    done_set = set(manifest.get("features_done", []))
    to_process = [u for u in uuids if u not in done_set]

    if not to_process:
        print(f"  [features] All {len(uuids)} maps already done.")
        return stats

    print(f"  [features] Processing {len(to_process)} maps ({len(uuids) - len(to_process)} already done)...")

    if workers <= 1:
        # Single-process (simpler, avoids multiprocessing overhead for fast ops)
        for i, uuid in enumerate(to_process, 1):
            result = extract_features_for_map(uuid)
            if result["status"] == "success":
                stats["success"] += 1
                stats["total_diffs"] += len(result["difficulties"])
                manifest["features_done"].append(uuid)
            elif result["status"] == "skipped":
                stats["skipped"] += 1
                manifest["features_done"].append(uuid)
            else:
                stats["failed"] += 1
                manifest["features_failed"].append(uuid)
                if stats["failed"] <= 3:
                    print(f"    FAIL {uuid}: {result['errors']}")

            if i % 100 == 0 or i == len(to_process):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                print(f"    [{i}/{len(to_process)}] OK={stats['success']} SKIP={stats['skipped']} "
                      f"FAIL={stats['failed']} | {rate:.0f} maps/s")
    else:
        # Multiprocessing for larger batches
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(extract_features_for_map, u): u for u in to_process}
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                result = future.result()
                if result["status"] == "success":
                    stats["success"] += 1
                    stats["total_diffs"] += len(result["difficulties"])
                    manifest["features_done"].append(result["uuid"])
                elif result["status"] == "skipped":
                    stats["skipped"] += 1
                    manifest["features_done"].append(result["uuid"])
                else:
                    stats["failed"] += 1
                    manifest["features_failed"].append(result["uuid"])

                if done_count % 100 == 0 or done_count == len(to_process):
                    elapsed = time.time() - t0
                    rate = done_count / elapsed if elapsed > 0 else 0
                    print(f"    [{done_count}/{len(to_process)}] OK={stats['success']} "
                          f"SKIP={stats['skipped']} FAIL={stats['failed']} | {rate:.0f} maps/s")

    stats["time_s"] = time.time() - t0
    return stats


def run_audio_batch(uuids: list[str], manifest: dict, workers: int = 1) -> dict:
    """Run audio mel extraction on a batch of UUIDs."""
    stats = {"success": 0, "no_audio": 0, "failed": 0, "exists": 0}
    t0 = time.time()

    done_set = set(manifest.get("audio_done", []))
    to_process = [u for u in uuids if u not in done_set]

    if not to_process:
        print(f"  [audio] All {len(uuids)} maps already done.")
        return stats

    print(f"  [audio] Processing {len(to_process)} maps ({len(uuids) - len(to_process)} already done)...")

    # Audio extraction is CPU-heavy (librosa), single-process is fine
    # multiprocessing actually hurts due to librosa's internal threading
    for i, uuid in enumerate(to_process, 1):
        result = extract_audio_for_map(uuid)

        if result["status"] == "success":
            stats["success"] += 1
            manifest["audio_done"].append(uuid)
        elif result["status"] == "exists":
            stats["exists"] += 1
            manifest["audio_done"].append(uuid)
        elif result["status"] == "no_audio":
            stats["no_audio"] += 1
            # Don't re-attempt maps without audio
            manifest["audio_done"].append(uuid)
        else:
            stats["failed"] += 1
            manifest["audio_failed"].append(uuid)
            if stats["failed"] <= 3:
                print(f"    FAIL {uuid}: {result.get('error', '?')}")

        if i % 20 == 0 or i == len(to_process):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(to_process) - i) / rate / 60 if rate > 0 else 0
            print(f"    [{i}/{len(to_process)}] OK={stats['success']} "
                  f"EXISTS={stats['exists']} NO_OGG={stats['no_audio']} "
                  f"FAIL={stats['failed']} | {rate:.1f} maps/s  ETA={eta:.1f}m")

    stats["time_s"] = time.time() - t0
    return stats


def get_disk_usage() -> dict:
    """Get current disk usage for dataset dirs."""
    import subprocess
    result = {}
    for name, path in [("features", FEATURES_DIR), ("audio_features", AUDIO_DIR)]:
        r = subprocess.run(["du", "-sh", str(path)], capture_output=True, text=True)
        if r.returncode == 0:
            result[name] = r.stdout.split()[0]
    return result


def main():
    parser = argparse.ArgumentParser(description="SynthRiders Batch Feature Extraction Engine")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Number of maps per batch (default: 500)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from manifest_progress.json")
    parser.add_argument("--skip-audio", action="store_true",
                        help="Skip audio mel extraction")
    parser.add_argument("--skip-features", action="store_true",
                        help="Skip beatmap feature extraction")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers for feature extraction")
    parser.add_argument("--offset", type=int, default=0,
                        help="Start from this map index (for manual batching)")
    args = parser.parse_args()

    # Ensure output dirs exist
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # Load manifest
    manifest = load_manifest() if args.resume else load_manifest()

    # Get all UUIDs
    all_uuids = get_all_uuids()
    manifest["total_maps"] = len(all_uuids)

    print("=" * 60)
    print("  SYNTHRIDERS EXTRACTION ENGINE")
    print("=" * 60)
    print(f"  Total corpus:     {len(all_uuids)} maps")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Offset:           {args.offset}")
    print(f"  Skip features:    {args.skip_features}")
    print(f"  Skip audio:       {args.skip_audio}")
    print(f"  Workers:          {args.workers}")
    print(f"  Features done:    {len(manifest.get('features_done', []))}")
    print(f"  Audio done:       {len(manifest.get('audio_done', []))}")
    print()

    # Select batch
    batch_uuids = all_uuids[args.offset : args.offset + args.batch_size]
    if not batch_uuids:
        print("[!] No maps to process at this offset.")
        return

    print(f"  Processing batch: maps {args.offset} to {args.offset + len(batch_uuids) - 1}")
    print(f"  Batch UUIDs: {batch_uuids[0]} ... {batch_uuids[-1]}")
    print()

    # Workstream A: Feature extraction
    if not args.skip_features:
        print("[WORKSTREAM A] Beatmap Feature Extraction")
        print("-" * 40)
        feat_stats = run_feature_batch(batch_uuids, manifest, workers=args.workers)
        save_manifest(manifest)
        print(f"  Features: {feat_stats.get('success', 0)} OK, "
              f"{feat_stats.get('skipped', 0)} skipped, "
              f"{feat_stats.get('failed', 0)} failed, "
              f"{feat_stats.get('total_diffs', 0)} difficulty files written")
        if "time_s" in feat_stats:
            print(f"  Time: {feat_stats['time_s']:.1f}s")
        print()

    # Workstream B: Audio mel extraction
    if not args.skip_audio:
        print("[WORKSTREAM B] Audio Mel Spectrogram Extraction")
        print("-" * 40)
        audio_stats = run_audio_batch(batch_uuids, manifest, workers=args.workers)
        save_manifest(manifest)
        print(f"  Audio: {audio_stats.get('success', 0)} OK, "
              f"{audio_stats.get('exists', 0)} existed, "
              f"{audio_stats.get('no_audio', 0)} no OGG, "
              f"{audio_stats.get('failed', 0)} failed")
        if "time_s" in audio_stats:
            print(f"  Time: {audio_stats['time_s']:.1f}s")
        print()

    # Summary report
    disk = get_disk_usage()
    feat_count = len([f for f in FEATURES_DIR.iterdir() if f.suffix == ".npz"])
    audio_count = len([f for f in AUDIO_DIR.iterdir() if f.suffix == ".npz"])

    print("=" * 60)
    print("  BATCH REPORT")
    print("=" * 60)
    print(f"  Total feature files:  {feat_count}")
    print(f"  Total audio files:    {audio_count}")
    print(f"  Feature disk usage:   {disk.get('features', '?')}")
    print(f"  Audio disk usage:     {disk.get('audio_features', '?')}")
    print(f"  Manifest features:    {len(manifest.get('features_done', []))}/{len(all_uuids)}")
    print(f"  Manifest audio:       {len(manifest.get('audio_done', []))}/{len(all_uuids)}")
    remaining = len(all_uuids) - len(set(manifest.get("features_done", [])))
    print(f"  Remaining (features): {remaining}")
    remaining_audio = len(all_uuids) - len(set(manifest.get("audio_done", [])))
    print(f"  Remaining (audio):    {remaining_audio}")
    print("=" * 60)

    save_manifest(manifest)
    print(f"\n[+] Manifest saved to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
