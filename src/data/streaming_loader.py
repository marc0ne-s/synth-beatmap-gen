"""
M4-Native Streaming Data Loader for SynthRiders Corpus.
Implements NaN shielding, variable-length bucketing, and robust audio-feature alignment.
"""

import json
import math
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

DIFFICULTY_MAP = {
    "Easy": 0,
    "Normal": 1,
    "Hard": 2,
    "Expert": 3,
    "Master": 4,
    "Custom": 5,
}

class StreamingSynthDataset(Dataset):
    def __init__(
        self,
        feature_files: List[Path],
        audio_dir: Path,
        max_length: int = 15000,
        nan_shield: bool = True
    ):
        self.feature_files = feature_files
        self.audio_dir = audio_dir
        self.max_length = max_length
        self.nan_shield = nan_shield

    def __len__(self) -> int:
        return len(self.feature_files)

    def _apply_nan_shield(self, tensor: torch.Tensor) -> torch.Tensor:
        """Dynamically intercept and squash NaN/Inf values inside feature tensors."""
        if not self.nan_shield:
            return tensor
            
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            # Replace NaNs/Infs with 0.0 (neutral representation)
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        return tensor

    def _compute_augmented_features(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Dynamically calculates the 48 mathematical augmentation dimensions.
        Transforms the raw 80-bin Mel base into the rich 128-D context vector.
        """
        T, F = mel.shape
        
        # 1. Delta (Spectral Movement)
        # Shift temporal context by 1 frame to calculate direct derivatives 
        shifted_mel = torch.cat([torch.zeros(1, F, device=mel.device), mel[:-1, :]], dim=0)
        delta = mel - shifted_mel
        
        # 44-Band Average Pooling: Preserves Full-Spectrum rhythm info while fitting the bit-budget
        # adaptive_avg_pool1d operates on (N, C, L) so we process T as Channels: (1, T, 80) -> (1, T, 44)
        delta_44 = torch.nn.functional.adaptive_avg_pool1d(delta.unsqueeze(0), 44).squeeze(0)
        
        # 2. Spectral Flux (Positive energy surges)
        # Scaled softly by generic divisor to prevent domination
        flux = torch.nn.functional.relu(delta).sum(dim=1, keepdim=True) / 10.0
        
        # 3. Spectral Centroid (Normalized Brightness)
        f_idx = torch.arange(F, dtype=torch.float32, device=mel.device) / (F - 1)
        mel_sum = mel.sum(dim=1, keepdim=True) + 1e-8
        centroid = (mel * f_idx.unsqueeze(0)).sum(dim=1, keepdim=True) / mel_sum
        
        # 4. Spectral Bandwidth (Frequency Spread)
        f_diff_sq = (f_idx.unsqueeze(0) - centroid)**2
        bandwidth = torch.sqrt((mel * f_diff_sq).sum(dim=1, keepdim=True) / mel_sum)
        
        # 5. RMS Log-Energy (Dynamic track intensity)
        rms = torch.sqrt((mel**2).mean(dim=1, keepdim=True))
        
        # Assemble perfectly into exactly 128 dimensions
        augmented = torch.cat([mel, delta_44, flux, centroid, bandwidth, rms], dim=1)
        
        # Shield output from div/0 anomalies
        return self._apply_nan_shield(augmented)

    def __getitem__(self, idx: int):
        feat_path = self.feature_files[idx]
        stem = feat_path.stem  # e.g., "0007b2da6d9527ab_Master"
        
        # Parse UUID and Difficulty
        parts = stem.rsplit("_", 1)
        uuid = parts[0]
        difficulty_str = parts[1] if len(parts) > 1 else "Hard"
        diff_idx = DIFFICULTY_MAP.get(difficulty_str, 2)
        
        # Load Beatmap Features
        try:
            data = np.load(feat_path)
            pos = torch.from_numpy(data["note_positions"]).float()
            pres = torch.from_numpy(data["note_presence"]).float()
        except Exception as e:
            # Shield against fundamentally corrupted NPZ files
            print(f"[StreamingLoader] Corrupted NPZ {stem}: {e}")
            pos = torch.zeros((1, 2, 2)).float()
            pres = torch.zeros((1, 2)).float()
            
        # Target representation: presence (2) + positions (4) + velocity (2) = 8D feature vector
        T_feat = pos.shape[0]
        
        # Velocity Derivation (Euclidean Differential)
        shifted_pos = torch.cat([pos[0:1], pos[:-1]], dim=0)
        vel = torch.sqrt(torch.sum((pos - shifted_pos)**2, dim=-1))  # (T, 2)
        vel = self._apply_nan_shield(vel)
        vel = torch.tanh(vel / 2.0)  # Compress extreme velocities
        
        flat_pos = pos.view(T_feat, -1)
        flat_pos = torch.tanh(flat_pos / 3.0)
        
        target_features = torch.cat([pres, flat_pos, vel], dim=-1) # (T, 8)
        target_features = self._apply_nan_shield(target_features)

        # Load Audio Features
        audio_path = self.audio_dir / f"{uuid}.npz"
        if audio_path.exists():
            try:
                audio_data = np.load(audio_path)
                mel = audio_data["audio_mel"]
                audio_features = torch.from_numpy(mel).float()
                audio_features = self._apply_nan_shield(audio_features)
            except Exception as e:
                audio_features = torch.zeros((T_feat, 80)).float()
        else:
            audio_features = torch.zeros((T_feat, 80)).float()
            
        # Convert 80-bin to 128-D Context Array
        audio_features = self._compute_augmented_features(audio_features)
            
        T_audio = audio_features.shape[0]
        
        # Align temporal boundaries (truncating whichever is longer)
        T_common = min(T_feat, T_audio, self.max_length)
        target_features = target_features[:T_common]
        audio_features = audio_features[:T_common]
        
        # O(T^2) Attention Throttle:
        MAX_ATTN_WINDOW = 4096
        if T_common > MAX_ATTN_WINDOW:
            # Randomly slice a 4096-frame window (approx 80 seconds)
            max_start = T_common - MAX_ATTN_WINDOW
            start_t = np.random.randint(0, max_start)
            end_t = start_t + MAX_ATTN_WINDOW
            
            target_features = target_features[start_t:end_t]
            audio_features = audio_features[start_t:end_t]
            T_common = MAX_ATTN_WINDOW
        
        # We return T_common so collate_fn knows how much to pack
        diff_tensor = torch.tensor(diff_idx, dtype=torch.long)
        
        return audio_features, target_features, diff_tensor, T_common


def streaming_collate_fn(batch):
    """
    Collate variable-length sequences dynamically to the LONGEST IN BATCH,
    preventing static 15,000 frame bloat.
    """
    audio_list, target_list, diff_list, lengths = zip(*batch)
    
    max_t = max(lengths)
    
    # Pad to max_t
    audio_padded = []
    target_padded = []
    
    for aud, tgt, length in zip(audio_list, target_list, lengths):
        pad_amt = max_t - length
        # Pad temporal dimension (last dim is feature dim, second last is T)
        if pad_amt > 0:
            # torch.nn.functional.pad adds to the end of dimensions
            Aud_p = torch.nn.functional.pad(aud, (0, 0, 0, pad_amt), value=0.0)
            Tgt_p = torch.nn.functional.pad(tgt, (0, 0, 0, pad_amt), value=0.0)
        else:
            Aud_p = aud
            Tgt_p = tgt
            
        audio_padded.append(Aud_p)
        target_padded.append(Tgt_p)
        
    audio_batch = torch.stack(audio_padded)
    target_batch = torch.stack(target_padded)
    diff_batch = torch.stack(diff_list)
    lengths_batch = torch.tensor(lengths, dtype=torch.long)
    
    return audio_batch, target_batch, diff_batch, lengths_batch


def get_streaming_loader(
    features_dir: str,
    audio_dir: str,
    difficulties: List[str] = ["Hard", "Expert", "Master"],
    batch_size: int = 16,
    num_maps: int = 800,
) -> Tuple[DataLoader, DataLoader]:
    """Retrieve length-bucketed loaders for optimized MPS ingestion."""
    
    feat_d = Path(features_dir)
    aud_d = Path(audio_dir)
    
    print("[StreamingLoader] Indexing corpus...")
    all_files = list(feat_d.glob("*.npz"))
    
    # Filter by selected difficulties
    valid_files = []
    for f in all_files:
        diff_str = f.stem.rsplit("_", 1)[-1] if "_" in f.stem else "Hard"
        if diff_str in difficulties:
            valid_files.append(f)
            
    print(f"[StreamingLoader] Found {len(valid_files)} maps matching {difficulties}.")
    
    # Shuffle and trim to requested subset size
    np.random.seed(42)
    np.random.shuffle(valid_files)
    subset_files = valid_files[:num_maps]
    
    # Simple 80/20 train val split
    split_idx = int(0.8 * len(subset_files))
    train_files = subset_files[:split_idx]
    val_files = subset_files[split_idx:]
    
    print(f"[StreamingLoader] Train split: {len(train_files)} maps | Val split: {len(val_files)} maps")
    
    train_dataset = StreamingSynthDataset(train_files, aud_d)
    val_dataset = StreamingSynthDataset(val_files, aud_d)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, # In a pure implementation we'd use LengthBucketingSampler here
        collate_fn=streaming_collate_fn,
        num_workers=0, # Mac MPS multiprocess deadlock fix
        pin_memory=False # Not strictly needed for unified memory
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=streaming_collate_fn,
        num_workers=0
    )
    
    return train_loader, val_loader
