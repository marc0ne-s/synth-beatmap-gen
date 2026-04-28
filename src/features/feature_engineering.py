"""
Feature engineering for SynthRiders beatmaps.

Converts parsed beatmap JSON into model-ready tensors.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import torch


# Constants
FRAME_MS = 20.0  # 20ms per frame (~50fps, matching VR rhythm games)
MAX_DURATION_SEC = 300.0  # 5 minutes max song length
MAX_FRAMES = int(MAX_DURATION_SEC * 1000 / FRAME_MS)

# Spatial grid parameters
X_BINS = 16  # horizontal lanes
Y_BINS = 8   # vertical lanes
X_MIN, X_MAX = -0.55, 0.55
Y_MIN, Y_MAX = -0.45, 0.45

# Note type mapping
HAND_RIGHT = 0
HAND_LEFT = 1
# Note: raw type values can be 0,1,2,3 (2/3 are rail variants)
HAND_MAP = {0: 0, 2: 0, 1: 1, 3: 1}  # map to right/left


def time_to_frame(time_ms: float, frame_ms: float = FRAME_MS) -> int:
    """Convert millisecond timestamp to frame index."""
    return int(time_ms / frame_ms)


def discretize_position(x: float, y: float, x_bins: int = X_BINS, y_bins: int = Y_BINS) -> Tuple[int, int]:
    """Discretize continuous x,y into grid bins."""
    x_bin = int((x - X_MIN) / (X_MAX - X_MIN) * x_bins)
    y_bin = int((y - Y_MIN) / (Y_MAX - Y_MIN) * y_bins)
    x_bin = max(0, min(x_bins - 1, x_bin))
    y_bin = max(0, min(y_bins - 1, y_bin))
    return x_bin, y_bin


def extract_beatmap_features(
    parsed_path: str,
    difficulty: str = "Hard",
    frame_ms: float = FRAME_MS,
    max_duration_sec: float = MAX_DURATION_SEC,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Extract model-ready features from a parsed beatmap JSON.

    Returns dict with:
        - note_occupancy: (T, 2, X_BINS, Y_BINS) binary tensor per hand
        - note_positions: (T, 2, 2) continuous x,y per hand (or -1 for no note)
        - note_times: (T,) frame timestamps in ms
        - rail_mask: (T,) binary, whether a rail starts at this frame
        - rail_paths: list of rail segment arrays (variable length)
        - metadata: dict with bpm, note_count, duration_ms
    """
    with open(parsed_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("status") != "success":
        return None

    diff_data = data.get("difficulties", {}).get(difficulty)
    if not diff_data or diff_data.get("note_count", 0) == 0:
        return None

    notes = diff_data["notes"]
    metadata = data.get("metadata", {})
    bpm = metadata.get("bpm", 120.0)

    # Determine song duration
    if notes:
        max_time = max(n["time"] for n in notes)
        duration_ms = min(max_time + 5000, max_duration_sec * 1000)
    else:
        duration_ms = max_duration_sec * 1000

    num_frames = time_to_frame(duration_ms, frame_ms) + 1
    num_frames = min(num_frames, MAX_FRAMES)

    # Initialize tensors
    note_occupancy = np.zeros((num_frames, 2, X_BINS, Y_BINS), dtype=np.float32)
    note_positions = np.full((num_frames, 2, 2), -1.0, dtype=np.float32)
    note_presence = np.zeros((num_frames, 2), dtype=np.float32)
    rail_mask = np.zeros(num_frames, dtype=np.float32)
    rail_paths: List[np.ndarray] = []

    for note in notes:
        t = float(note["time"])
        frame = time_to_frame(t, frame_ms)
        if frame >= num_frames:
            continue

        hand = HAND_MAP.get(note["type"], 0)  # map 0,2->0 (right), 1,3->1 (left)
        x, y = float(note["x"]), float(note["y"])
        if math.isnan(x) or math.isnan(y):
            continue

        # Discretize position
        x_bin, y_bin = discretize_position(x, y)
        note_occupancy[frame, hand, x_bin, y_bin] = 1.0

        # Continuous position
        note_positions[frame, hand, 0] = x
        note_positions[frame, hand, 1] = y
        note_presence[frame, hand] = 1.0

        # Rail segments
        segments = note.get("segments")
        if segments and isinstance(segments, list) and len(segments) > 0:
            rail_mask[frame] = 1.0
            rail_paths.append(np.array(segments, dtype=np.float32))

    return {
        "note_occupancy": note_occupancy,
        "note_positions": note_positions,
        "note_presence": note_presence,
        "rail_mask": rail_mask,
        "rail_paths": rail_paths,
        "note_times": np.arange(num_frames, dtype=np.float32) * frame_ms,
        "metadata": {
            "bpm": bpm,
            "note_count": len(notes),
            "duration_ms": duration_ms,
            "duration_frames": num_frames,
            "difficulty": difficulty,
        },
    }


def save_features(
    features: Dict[str, np.ndarray],
    output_path: str,
) -> None:
    """Save features to compressed NPZ."""
    np.savez_compressed(
        output_path,
        note_occupancy=features["note_occupancy"],
        note_positions=features["note_positions"],
        note_presence=features["note_presence"],
        rail_mask=features["rail_mask"],
        note_times=features["note_times"],
        bpm=features["metadata"]["bpm"],
        note_count=features["metadata"]["note_count"],
        duration_ms=features["metadata"]["duration_ms"],
        duration_frames=features["metadata"]["duration_frames"],
        difficulty=features["metadata"]["difficulty"],
    )


def load_features(path: str) -> Dict[str, np.ndarray]:
    """Load features from NPZ."""
    data = np.load(path)
    return {
        "note_occupancy": data["note_occupancy"],
        "note_positions": data["note_positions"],
        "note_presence": data["note_presence"],
        "rail_mask": data["rail_mask"],
        "note_times": data["note_times"],
        "metadata": {
            "bpm": float(data["bpm"]),
            "note_count": int(data["note_count"]),
            "duration_ms": float(data["duration_ms"]),
            "duration_frames": int(data["duration_frames"]),
            "difficulty": str(data["difficulty"]),
        },
    }


class SynthBeatmapDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for SynthRiders beatmap features.

    Loads real audio mel spectrograms when available, falling back to
    synthetic random features for maps without extracted audio.
    """

    def __init__(
        self,
        features_dir: str,
        difficulty: str = "Hard",
        max_length: int = MAX_FRAMES,
        audio_features_dir: Optional[str] = None,
    ):
        self.features_dir = Path(features_dir)
        self.difficulty = difficulty
        self.max_length = max_length
        self.audio_features_dir = Path(audio_features_dir) if audio_features_dir else None

        # Find all .npz feature files
        self.files = sorted(self.features_dir.glob("*.npz"))
        if not self.files:
            raise ValueError(f"No .npz files found in {features_dir}")

        # Filter by difficulty
        self.valid_files = []
        for f in self.files:
            data = np.load(f)
            if str(data.get("difficulty", "")) == difficulty:
                self.valid_files.append(f)

        self.audio_hits = 0
        self.audio_misses = 0

        print(f"[Dataset] {len(self.valid_files)}/{len(self.files)} files match difficulty '{difficulty}'")
        if self.audio_features_dir:
            print(f"[Dataset] Audio features dir: {self.audio_features_dir}")

    def __len__(self) -> int:
        return len(self.valid_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            audio_features: (T, F) real or synthetic audio features
            note_occupancy: (T, 2, X, Y) binary
            note_positions: (T, 2, 2) continuous x,y
        """
        data = np.load(self.valid_files[idx])

        occ = torch.from_numpy(data["note_occupancy"]).float()
        pos = torch.from_numpy(data["note_positions"]).float()
        pres = torch.from_numpy(data["note_presence"]).float()

        T = occ.shape[0]

        # Pad or truncate to max_length
        if T < self.max_length:
            pad = self.max_length - T
            occ = torch.nn.functional.pad(occ, (0, 0, 0, 0, 0, 0, 0, pad))
            pos = torch.nn.functional.pad(pos, (0, 0, 0, 0, 0, pad), value=-1.0)
            pres = torch.nn.functional.pad(pres, (0, 0, 0, pad))
        else:
            occ = occ[: self.max_length]
            pos = pos[: self.max_length]
            pres = pres[: self.max_length]
            T = self.max_length

        # Try to load real audio features
        audio_features = None
        if self.audio_features_dir:
            uuid = self.valid_files[idx].stem
            audio_path = self.audio_features_dir / f"{uuid}.npz"
            if audio_path.exists():
                audio_data = np.load(audio_path)
                mel = audio_data["audio_mel"]
                if mel.shape[0] >= T:
                    audio_features = torch.from_numpy(mel[:T]).float()
                    self.audio_hits += 1
                else:
                    # Pad short audio
                    pad = T - mel.shape[0]
                    audio_features = torch.from_numpy(mel).float()
                    audio_features = torch.nn.functional.pad(audio_features, (0, 0, 0, pad))
                    self.audio_hits += 1

        if audio_features is None:
            # Fallback to synthetic features
            audio_features = torch.randn(T, 80)
            self.audio_misses += 1

        # Pad or truncate audio features to max_length
        if audio_features.shape[0] < self.max_length:
            pad = self.max_length - audio_features.shape[0]
            audio_features = torch.nn.functional.pad(audio_features, (0, 0, 0, pad))
        else:
            audio_features = audio_features[: self.max_length]

        return audio_features, occ, pos, pres, T


def batch_process_features(
    parsed_dir: str,
    output_dir: str,
    difficulty: str = "Hard",
) -> None:
    """Batch process all parsed beatmaps into features."""
    os.makedirs(output_dir, exist_ok=True)

    parsed_files = sorted(Path(parsed_dir).glob("*.json"))
    total = len(parsed_files)
    success = 0
    skipped = 0

    print(f"[+] Processing {total} parsed beatmaps into features...")

    for i, parsed_path in enumerate(parsed_files, 1):
        try:
            features = extract_beatmap_features(str(parsed_path), difficulty=difficulty)
            if features is None:
                skipped += 1
                continue

            out_name = parsed_path.stem + ".npz"
            out_path = os.path.join(output_dir, out_name)
            save_features(features, out_path)
            success += 1

            if i % 500 == 0 or i == total:
                print(f"[{i}/{total}] OK={success} SKIP={skipped}")

        except Exception as e:
            print(f"ERR {parsed_path.name}: {e}")

    print(f"[+] Done: {success} features created, {skipped} skipped")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract features from parsed beatmaps")
    parser.add_argument("--parsed-dir", default="/Volumes/Second-Brain-1/AI/Synth/dataset/parsed")
    parser.add_argument("--output-dir", default="/Volumes/Second-Brain-1/AI/Synth/dataset/features")
    parser.add_argument("--difficulty", default="Hard")
    args = parser.parse_args()

    batch_process_features(args.parsed_dir, args.output_dir, args.difficulty)
