#!/usr/bin/env python3
"""
batch_extract.py — Batch decrypt and extract .synth files for ML ingestion.

Usage:
    python batch_extract.py --raw-dir dataset/raw --out-dir dataset/extracted --password "..."

Produces:
    dataset/extracted/<uuid>/        # One dir per map
        track.data.json             # Song metadata (BPM, title, artist...)
        beatmap.meta.bin            # Binary note data
        cover.jpg                   # Album art
        audio.ogg                   # Audio track
        synthriderz.meta.json       # Community metadata
    dataset/index.json              # Master index of all maps
    dataset/reports/failed.log      # List of failed extractions
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional


def generate_uuid_from_path(path: str) -> str:
    """Deterministic UUID from full file path."""
    return hashlib.sha256(path.encode("utf-8")).hexdigest()[:16]


def sanitize_filename(name: str) -> str:
    """Sanitize for filesystem use."""
    keep = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
    return "".join(c if c in keep else "_" for c in name)


def extract_synth(
    synth_path: str,
    out_dir: str,
    password: str,
) -> Optional[dict]:
    """
    Extract a single .synth file to a directory.
    Returns metadata dict or None on failure.
    """
    try:
        import pyzipper
    except ImportError:
        raise RuntimeError("pyzipper is required. Install: pip install pyzipper")

    file_uuid = generate_uuid_from_path(synth_path)
    extract_dir = os.path.join(out_dir, file_uuid)
    os.makedirs(extract_dir, exist_ok=True)

    try:
        with pyzipper.AESZipFile(synth_path, "r") as zf:
            zf.setpassword(password.encode("utf-8"))
            zf.extractall(extract_dir)
    except Exception as e:
        return {"status": "failed", "error": str(e), "path": synth_path, "uuid": file_uuid}

    # Parse track.data.json for metadata
    track_data = {}
    track_path = os.path.join(extract_dir, "track.data.json")
    if os.path.exists(track_path):
        try:
            with open(track_path, "rb") as f:
                raw = f.read()
                # Handle UTF-16 BOM
                if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
                    text = raw.decode("utf-16")
                else:
                    text = raw.decode("utf-8")
                track_data = json.loads(text)
        except Exception as e:
            track_data = {"_parse_error": str(e)}

    # Get file sizes
    beatmap_size = 0
    beatmap_path = os.path.join(extract_dir, "beatmap.meta.bin")
    if os.path.exists(beatmap_path):
        beatmap_size = os.path.getsize(beatmap_path)

    cover_file = None
    audio_file = None
    for f in os.listdir(extract_dir):
        f_lower = f.lower()
        if f_lower.endswith((".jpg", ".jpeg", ".png")) and "cover" not in f_lower:
            # Rename to cover.jpg for consistency
            old = os.path.join(extract_dir, f)
            new = os.path.join(extract_dir, "cover.jpg")
            if old != new and not os.path.exists(new):
                os.rename(old, new)
            cover_file = "cover.jpg"
        elif f_lower.endswith(".ogg"):
            old = os.path.join(extract_dir, f)
            new = os.path.join(extract_dir, "audio.ogg")
            if old != new and not os.path.exists(new):
                os.rename(old, new)
            audio_file = "audio.ogg"

    return {
        "status": "success",
        "uuid": file_uuid,
        "path": synth_path,
        "extract_dir": extract_dir,
        "song_title": track_data.get("name", ""),
        "artist": track_data.get("artist", ""),
        "bpm": track_data.get("bpm", None),
        "difficulty": track_data.get("supportedDifficulties", []),
        "mapper": track_data.get("mapper", ""),
        "duration": track_data.get("duration", ""),
        "beatmap_size": beatmap_size,
        "files": os.listdir(extract_dir),
    }


def batch_extract(raw_dir: str, out_dir: str, password: str, report_dir: str) -> None:
    synth_files = sorted([
        os.path.join(raw_dir, f)
        for f in os.listdir(raw_dir)
        if f.endswith(".synth")
    ])

    total = len(synth_files)
    print(f"[+] Found {total} .synth files to extract")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    index = []
    failed = []
    start_time = time.time()

    for i, synth_path in enumerate(synth_files, 1):
        result = extract_synth(synth_path, out_dir, password)
        if result is None:
            continue

        if result["status"] == "success":
            index.append(result)
        else:
            failed.append(result)

        if i % 100 == 0 or i == total:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate if rate > 0 else 0
            print(
                f"[{i}/{total}] OK={len(index)} FAIL={len(failed)} "
                f"rate={rate:.1f}f/s eta={eta/60:.1f}m"
            )

    # Save index
    index_path = os.path.join(os.path.dirname(out_dir), "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_files": total,
            "success": len(index),
            "failed": len(failed),
            "maps": index,
        }, f, indent=2, ensure_ascii=False)
    print(f"[+] Saved index to {index_path}")

    # Save failed log
    if failed:
        failed_path = os.path.join(report_dir, "failed.log")
        with open(failed_path, "w", encoding="utf-8") as f:
            for item in failed:
                f.write(f"{item['path']} | {item['error']}\n")
        print(f"[+] Saved failed log to {failed_path} ({len(failed)} failures)")

    print(f"[+] Done: {len(index)} succeeded, {len(failed)} failed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch extract .synth files")
    parser.add_argument("--raw-dir", default="dataset/raw", help="Directory with .synth files")
    parser.add_argument("--out-dir", default="dataset/extracted", help="Output directory")
    parser.add_argument("--password", default="hC2*wE5R*qQzv@a!", help="AES-256 ZIP password")
    parser.add_argument("--report-dir", default="dataset/reports", help="Report directory")
    args = parser.parse_args()

    base = "/Volumes/Second-Brain-1/AI/Synth"
    raw_dir = os.path.join(base, args.raw_dir) if not os.path.isabs(args.raw_dir) else args.raw_dir
    out_dir = os.path.join(base, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    report_dir = os.path.join(base, args.report_dir) if not os.path.isabs(args.report_dir) else args.report_dir

    batch_extract(raw_dir, out_dir, args.password, report_dir)


if __name__ == "__main__":
    main()
