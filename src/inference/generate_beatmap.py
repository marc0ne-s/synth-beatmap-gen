#!/usr/bin/env python3
"""
Inference pipeline: generate a SynthRiders beatmap from audio.

Usage:
    python generate_beatmap.py \
        --checkpoint models/checkpoints/best_model.pt \
        --audio path/to/song.ogg \
        --output path/to/output.synth \
        --difficulty Hard

Or from existing feature NPZ:
    python generate_beatmap.py \
        --checkpoint models/checkpoints/best_model.pt \
        --feature-npz path/to/features.npz \
        --output path/to/output.synth
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.audio.audio_features import extract_librosa_features
from src.features.feature_engineering import FRAME_MS, X_BINS, X_MAX, X_MIN, Y_BINS, Y_MAX, Y_MIN
from src.models.baseline import BaselineBeatmapModel


def load_model(checkpoint_path: str, device: torch.device) -> BaselineBeatmapModel:
    """Load trained model from checkpoint."""
    model = BaselineBeatmapModel(audio_features=80)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model


def extract_audio_features(audio_path: str) -> np.ndarray:
    """Extract mel spectrogram from audio file, resample to 20ms frames."""
    features = extract_librosa_features(audio_path)
    mel = features["mel"]  # (T_audio, n_mels)

    # Resample to 20ms frame resolution
    sr = 22050
    hop_length = 512
    frame_duration_s = FRAME_MS / 1000.0
    audio_duration_s = mel.shape[0] * (hop_length / sr)
    target_frames = int(audio_duration_s / frame_duration_s)

    import torch
    import torch.nn.functional as F

    mel_t = torch.from_numpy(mel).unsqueeze(0).permute(0, 2, 1)  # (1, n_mels, T_audio)
    mel_t = F.interpolate(mel_t, size=target_frames, mode="linear", align_corners=False)
    mel_t = mel_t.squeeze(0).permute(1, 0)  # (target_frames, n_mels)
    return mel_t.numpy()


def predict_from_audio(
    model: BaselineBeatmapModel,
    audio_features: np.ndarray,
    device: torch.device,
    threshold: float = 0.5,
) -> list[dict]:
    """Run model inference and convert predictions to note list."""
    # audio_features: (T, 80)
    audio_t = torch.from_numpy(audio_features).unsqueeze(0).float().to(device)  # (1, T, 80)

    with torch.no_grad():
        preds = model(audio_t)

    presence_logits = preds["presence_logits"].squeeze(0).cpu().numpy()  # (T, 2)
    position_pred = preds["position_pred"].squeeze(0).cpu().numpy()  # (T, 4)

    presence_prob = 1.0 / (1.0 + np.exp(-presence_logits))  # sigmoid

    notes = []
    for t in range(presence_prob.shape[0]):
        for hand in range(2):
            if presence_prob[t, hand] > threshold:
                time_ms = float(t * FRAME_MS)
                x = float(position_pred[t, hand * 2])
                y = float(position_pred[t, hand * 2 + 1])

                # Clamp to valid ranges
                x = max(X_MIN, min(X_MAX, x))
                y = max(Y_MIN, min(Y_MAX, y))

                notes.append({
                    "time": time_ms,
                    "type": int(hand),  # 0=right, 1=left
                    "x": x,
                    "y": y,
                    "z": 0.0,  # Hit plane
                    "direction": [0.0, 1.0],  # Default upward
                    "presence_prob": float(presence_prob[t, hand]),
                })

    return notes


def deduplicate_notes(notes: list[dict], min_gap_ms: float = 150.0) -> list[dict]:
    """Remove notes that are too close in time for the same hand."""
    if not notes:
        return []

    # Sort by hand then time
    notes_sorted = sorted(notes, key=lambda n: (n["type"], n["time"]))
    filtered = [notes_sorted[0]]

    for note in notes_sorted[1:]:
        last = filtered[-1]
        if note["type"] == last["type"] and (note["time"] - last["time"]) < min_gap_ms:
            continue
        filtered.append(note)

    return filtered


def generate_synth_dict(notes: list[dict], metadata: dict) -> dict:
    """Generate a SynthRiders-compatible beatmap dict."""
    return {
        "version": "2.0",
        "metadata": {
            "songName": metadata.get("song_name", "AI Generated"),
            "authorName": metadata.get("author", "SynthRiders AI"),
            "difficulty": metadata.get("difficulty", "Hard"),
            "bpm": metadata.get("bpm", 128.0),
            "offset": metadata.get("offset", 0.0),
        },
        "notes": notes,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate beatmap from audio")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--audio", default=None, help="Path to audio file (OGG/WAV)")
    parser.add_argument("--feature-npz", default=None, help="Path to precomputed feature NPZ")
    parser.add_argument("--output", required=True, help="Output .synth JSON path")
    parser.add_argument("--difficulty", default="Hard")
    parser.add_argument("--song-name", default="AI Generated")
    parser.add_argument("--author", default="SynthRiders AI")
    parser.add_argument("--bpm", type=float, default=128.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dedup-gap", type=float, default=150.0, help="Minimum gap between same-hand notes (ms)")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    print(f"[+] Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    print(f"[+] Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Get audio features
    if args.feature_npz:
        print(f"[+] Loading features from {args.feature_npz}")
        data = np.load(args.feature_npz)
        if "audio_mel" in data:
            audio_features = data["audio_mel"]
        else:
            # Fallback to synthetic
            audio_features = np.random.randn(int(data["duration_ms"].item() / FRAME_MS) + 1, 80).astype(np.float32)
    elif args.audio:
        print(f"[+] Extracting features from {args.audio}")
        audio_features = extract_audio_features(args.audio)
    else:
        print("[!] Provide --audio or --feature-npz")
        return 1

    print(f"[+] Audio features: {audio_features.shape}")

    # Inference
    print("[+] Running inference...")
    notes = predict_from_audio(model, audio_features, device, threshold=args.threshold)
    print(f"[+] Raw predictions: {len(notes)} notes")

    # Post-process
    notes = deduplicate_notes(notes, min_gap_ms=args.dedup_gap)
    print(f"[+] After dedup: {len(notes)} notes")

    # Build output
    metadata = {
        "song_name": args.song_name,
        "author": args.author,
        "difficulty": args.difficulty,
        "bpm": args.bpm,
    }
    synth = generate_synth_dict(notes, metadata)

    # Write .synth file (encrypted ZIP with JSON inside)
    print("[+] Packaging .synth file...")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
    from synth_decryptor import write_synth

    # Load password from memory if available
    password = "hC2*wE5R*qQzv@a!"

    write_synth(
        synth_path=args.output,
        data=synth,
        password=password,
    )

    print(f"[+] Wrote {len(notes)} notes to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
