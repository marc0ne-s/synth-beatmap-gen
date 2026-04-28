"""
Extract real audio features from OGG files and pair with beatmap features.

Usage:
    python extract_audio_features.py --parsed-dir dataset/parsed --extracted-dir dataset/extracted --output-dir dataset/audio_features
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

# Lazy import librosa (heavy dependency)
librosa = None


def _get_librosa():
    global librosa
    if librosa is None:
        import librosa as lr
        librosa = lr
    return librosa


def extract_mel_from_ogg(
    ogg_path: str,
    target_frames: int,
    sr: int = 22050,
    n_mels: int = 80,
    hop_length: int = 512,
    frame_ms: float = 20.0,
) -> np.ndarray | None:
    """Extract mel spectrogram from OGG and resample to match beatmap frames.

    Args:
        ogg_path: Path to audio.ogg
        target_frames: Number of beatmap frames (20ms each)
        sr: Sample rate
        n_mels: Number of mel bins
        hop_length: STFT hop length
        frame_ms: Duration of each beatmap frame in ms

    Returns:
        mel: (target_frames, n_mels) float32, or None on failure
    """
    lr = _get_librosa()

    try:
        y, sr = lr.load(ogg_path, sr=sr, mono=True)
    except Exception as e:
        print(f"  Failed to load audio: {e}")
        return None

    # Compute mel spectrogram
    mel = lr.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_db = lr.power_to_db(mel, ref=np.max)  # (n_mels, T_audio)
    mel_db = mel_db.T.astype(np.float32)  # (T_audio, n_mels)

    # Resample to target_frames using linear interpolation
    if mel_db.shape[0] == target_frames:
        return mel_db

    # Use torch for interpolation (already a dependency)
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(mel_db).unsqueeze(0)  # (1, T_audio, n_mels)
    x = x.permute(0, 2, 1)  # (1, n_mels, T_audio)
    x = F.interpolate(
        x,
        size=target_frames,
        mode="linear",
        align_corners=False,
    )
    x = x.squeeze(0).permute(1, 0)  # (target_frames, n_mels)

    return x.numpy().astype(np.float32)


def batch_extract_audio_features(
    parsed_dir: str,
    extracted_dir: str,
    output_dir: str,
    difficulty: str = "Hard",
    n_mels: int = 80,
) -> None:
    """Extract mel spectrograms for all maps that have valid beatmap features."""
    os.makedirs(output_dir, exist_ok=True)

    parsed_files = sorted(Path(parsed_dir).glob("*.json"))
    total = len(parsed_files)
    success = 0
    skipped = 0
    errors = 0

    print(f"[+] Extracting audio features for {total} maps...")
    print(f"    Output: {output_dir}")
    print(f"    Mel bins: {n_mels}")
    print()

    t0 = time.time()

    for i, parsed_path in enumerate(parsed_files, 1):
        uuid = parsed_path.stem
        out_path = Path(output_dir) / f"{uuid}.npz"

        # Skip if already exists
        if out_path.exists():
            skipped += 1
            if i % 500 == 0:
                print(f"[{i}/{total}] SKIP(existing)={skipped} OK={success} ERR={errors}")
            continue

        # Load parsed JSON to get beatmap frame count and find OGG
        try:
            with open(parsed_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            errors += 1
            continue

        if data.get("status") != "success":
            skipped += 1
            continue

        diff_data = data.get("difficulties", {}).get(difficulty)
        if not diff_data or diff_data.get("note_count", 0) == 0:
            skipped += 1
            continue

        # Find OGG from parsed path field
        meta_path = data.get("path", "")
        if not meta_path:
            errors += 1
            continue

        # meta_path points to beatmap.meta.bin inside extracted dir
        ogg_path = Path(meta_path).parent / "audio.ogg"
        if not ogg_path.exists():
            errors += 1
            continue

        # Compute target frames from max note time
        notes = diff_data["notes"]
        if notes:
            max_time_ms = max(float(n["time"]) for n in notes)
            duration_ms = min(max_time_ms + 5000, 300000)
        else:
            duration_ms = 300000

        target_frames = int(duration_ms / 20.0) + 1
        target_frames = min(target_frames, 15000)  # cap at 5 minutes

        # Extract mel
        mel = extract_mel_from_ogg(str(ogg_path), target_frames, n_mels=n_mels)
        if mel is None:
            errors += 1
            continue

        # Save
        np.savez_compressed(out_path, audio_mel=mel, uuid=uuid, bpm=data.get("metadata", {}).get("bpm", 120.0))
        success += 1

        if i % 10 == 0 or i == total:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            print(f"[{i}/{total}] OK={success} SKIP={skipped} ERR={errors} | {rate:.1f} maps/s")

    print()
    print(f"[+] Done: {success} audio features extracted, {skipped} skipped, {errors} errors")
    print(f"    Total time: {time.time() - t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Extract audio features from OGG files")
    parser.add_argument("--parsed-dir", default="/Volumes/Second-Brain-1/AI/Synth/dataset/parsed")
    parser.add_argument("--extracted-dir", default="/Volumes/Second-Brain-1/AI/Synth/dataset/extracted")
    parser.add_argument("--output-dir", default="/Volumes/Second-Brain-1/AI/Synth/dataset/audio_features")
    parser.add_argument("--difficulty", default="Hard")
    parser.add_argument("--n-mels", type=int, default=80)
    args = parser.parse_args()

    batch_extract_audio_features(
        parsed_dir=args.parsed_dir,
        extracted_dir=args.extracted_dir,
        output_dir=args.output_dir,
        difficulty=args.difficulty,
        n_mels=args.n_mels,
    )


if __name__ == "__main__":
    main()
