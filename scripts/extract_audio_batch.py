#!/usr/bin/env python3
"""Minimal audio feature extractor for SynthRiders."""

import json
import os
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F

PARSED_DIR = Path("/Volumes/Second-Brain-1/AI/Synth/dataset/parsed")
OUTPUT_DIR = Path("/Volumes/Second-Brain-1/AI/Synth/dataset/audio_features")
DIFFICULTY = "Hard"
SR = 22050
N_MELS = 80
HOP_LENGTH = 512


def extract_mel(ogg_path: str, target_frames: int) -> np.ndarray:
    y, sr = librosa.load(ogg_path, sr=SR, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_t = torch.from_numpy(mel_db).unsqueeze(0)  # (1, n_mels, T_audio)
    mel_t = F.interpolate(mel_t, size=target_frames, mode="linear", align_corners=False)
    return mel_t.squeeze(0).permute(1, 0).numpy().astype(np.float32)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    parsed_files = sorted(PARSED_DIR.glob("*.json"))
    total = len(parsed_files)
    success = 0
    skipped = 0
    errors = 0
    t0 = time.time()

    print(f"[+] Processing {total} maps...")

    for i, parsed_path in enumerate(parsed_files, 1):
        uuid = parsed_path.stem
        out_path = OUTPUT_DIR / f"{uuid}.npz"
        if out_path.exists():
            skipped += 1
            if i % 100 == 0:
                print(f"  [{i}/{total}] skip(existing)={skipped}")
            continue

        with open(parsed_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("status") != "success":
            errors += 1
            continue

        diff = data.get("difficulties", {}).get(DIFFICULTY)
        if not diff or diff.get("note_count", 0) == 0:
            skipped += 1
            continue

        meta_path = data.get("path", "")
        if not meta_path:
            errors += 1
            continue

        ogg_path = Path(meta_path).parent / "audio.ogg"
        if not ogg_path.exists():
            errors += 1
            continue

        notes = diff["notes"]
        max_time_ms = max(float(n["time"]) for n in notes)
        duration_ms = min(max_time_ms + 5000, 300_000)
        target_frames = min(int(duration_ms / 20.0) + 1, 15_000)

        try:
            mel = extract_mel(str(ogg_path), target_frames)
        except Exception as e:
            print(f"  ERR {uuid}: {e}")
            errors += 1
            continue

        np.savez_compressed(out_path, audio_mel=mel, uuid=uuid,
                            bpm=data.get("metadata", {}).get("bpm", 120.0))
        success += 1

        if i % 50 == 0 or i == total:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            print(f"  [{i}/{total}] OK={success} SKIP={skipped} ERR={errors} | {rate:.1f} maps/s")

    print(f"[+] Done: {success} OK, {skipped} skipped, {errors} errors in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
