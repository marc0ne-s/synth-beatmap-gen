"""
Audio feature extraction pipeline for SynthRiders.

Designed for real audio files (WAV/OGG) but includes synthetic generator
for baseline overfit testing without matching audio.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch


def generate_synthetic_audio_features(
    num_frames: int,
    n_mels: int = 80,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate synthetic audio features for baseline overfit testing.

    Produces a deterministic "spectrogram-like" tensor that the model can
    overfit to, proving the decoder pipeline works.

    Returns:
        features: (num_frames, n_mels) float32
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Create a smooth base signal
    t = np.linspace(0, 1, num_frames)
    base = np.zeros((num_frames, n_mels), dtype=np.float32)

    # Random frequency bands
    for i in range(n_mels):
        freq = rng.uniform(1, 10)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.1, 1.0)
        base[:, i] = amp * np.sin(2 * np.pi * freq * t + phase)

    # Add harmonic structure
    for i in range(n_mels // 4):
        if rng.random() > 0.5:
            band = rng.integers(0, n_mels - 4)
            base[:, band : band + 4] += 0.3 * np.sin(2 * np.pi * rng.uniform(2, 8) * t)[:, None]

    # Normalize to [0, 1]
    base = (base - base.min()) / (base.max() - base.min() + 1e-8)

    return base


def extract_librosa_features(
    audio_path: str,
    sr: int = 22050,
    n_mels: int = 80,
    hop_length: int = 512,
) -> Dict[str, np.ndarray]:
    """
    Extract audio features using librosa.

    Returns dict with:
        - mel: (T, n_mels) mel spectrogram
        - chroma: (T, 12) chroma features
        - onset: (T,) onset strength envelope
        - tempogram: (T, 384) tempogram
    """
    import librosa

    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)

    # Onset envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Tempogram
    tempogram = librosa.feature.tempogram(y=y, sr=sr, hop_length=hop_length)

    # Transpose to (T, F)
    return {
        "mel": mel_db.T.astype(np.float32),
        "chroma": chroma.T.astype(np.float32),
        "onset": onset_env.astype(np.float32),
        "tempogram": tempogram.T.astype(np.float32),
    }


def audio_features_to_frames(
    audio_features: np.ndarray,
    target_frames: int,
    frame_ms: float = 20.0,
    hop_length: int = 512,
    sr: int = 22050,
) -> np.ndarray:
    """
    Resample audio features to match beatmap frame resolution.

    Args:
        audio_features: (T_audio, F) from librosa
        target_frames: number of beatmap frames desired

    Returns:
        (target_frames, F) resampled features
    """
    import torch.nn.functional as F

    # Convert to torch tensor
    x = torch.from_numpy(audio_features).unsqueeze(0).unsqueeze(0)  # (1, 1, T, F)

    # Interpolate to target length
    x = F.interpolate(
        x.permute(0, 3, 1, 2),  # (1, F, 1, T)
        size=target_frames,
        mode="linear",
        align_corners=False,
    )
    x = x.permute(0, 2, 3, 1).squeeze(0).squeeze(0)  # (target_frames, F)

    return x.numpy()


class AudioFeatureExtractor:
    """Unified audio feature extractor."""

    def __init__(
        self,
        sr: int = 22050,
        n_mels: int = 80,
        hop_length: int = 512,
    ):
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length

    def extract(self, audio_path: str) -> Dict[str, np.ndarray]:
        """Extract all features from an audio file."""
        return extract_librosa_features(
            audio_path,
            sr=self.sr,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
        )

    def get_mel_only(self, audio_path: str) -> np.ndarray:
        """Get just the mel spectrogram."""
        return self.extract(audio_path)["mel"]
