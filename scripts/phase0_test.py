#!/usr/bin/env python3
"""
Phase 0: End-to-end pipeline test.
Test audio file → AI generation → .synth export → validation.
"""

import os
import sys
import json
import shutil
import subprocess
import zipfile
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import librosa

ROOT = Path("/Volumes/Second-Brain-1/AI/Synth")
sys.path.insert(0, str(ROOT))

from src.models.transformer import TransformerCausalDecoder

# === CONSTANTS ===
TIME_SCALE = 20.0
INDEX_SCALE = 64
GRID_SCALE = 0.1365
X_OFFSET = 0.002
Y_OFFSET = 0.0012
NMS_WINDOW = 100.0
DETECTION_THRESHOLD = 0.5

# === AUDIO FEATURE EXTRACTION ===

def extract_features_from_audio(audio_path: str) -> tuple:
    """Extract mel features from raw audio matching the training pipeline."""
    print(f"[+] Loading audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"    Duration: {duration:.1f}s, SR: {sr}")
    
    # Compute mel spectrogram
    print("[+] Computing mel spectrogram...")
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Compute audio features per 100ms frame
    hop = int(sr * 0.1)  # 100ms hop
    frames = []
    for i in range(0, len(y) - hop, hop):
        frame = y[i:i + hop]
        frame_mel = librosa.feature.melspectrogram(
            y=frame, sr=sr, n_mels=128, n_fft=2048, hop_length=hop
        )
        frame_db = librosa.power_to_db(frame_mel, ref=np.max)
        # Mean across time for this frame
        frame_tensor = torch.from_numpy(frame_db).float().mean(dim=1)
        frames.append(frame_tensor)
    
    if not frames:
        raise RuntimeError("Audio too short")
    
    mel_per_frame = torch.stack(frames).unsqueeze(0)
    
    # Compute augmented features matching training
    T, num_freq = mel_per_frame.shape[1], mel_per_frame.shape[2]
    mel_sq = mel_per_frame[0]
    shifted = torch.cat([torch.zeros(1, num_freq), mel_sq[:-1, :]], dim=0)
    delta = mel_sq - shifted
    delta_cpu = delta.cpu().unsqueeze(0)
    delta_44 = F.adaptive_avg_pool1d(delta_cpu, 44).squeeze(0)
    flux = F.relu(delta).sum(dim=1, keepdim=True) / 10.0
    f_idx = torch.arange(num_freq, dtype=torch.float32) / (num_freq - 1)
    mel_sum = mel_sq.sum(dim=1, keepdim=True) + 1e-8
    centroid = (mel_sq * f_idx.unsqueeze(0)).sum(dim=1, keepdim=True) / mel_sum
    f_diff_sq = (f_idx.unsqueeze(0) - centroid)**2
    bandwidth = torch.sqrt((mel_sq * f_diff_sq).sum(dim=1, keepdim=True) / mel_sum)
    rms = torch.sqrt((mel_sq**2).mean(dim=1, keepdim=True))
    augmented = torch.cat([mel_sq, delta_44, flux, centroid, bandwidth, rms], dim=1)
    
    return augmented.unsqueeze(0), duration

# === MODEL INFERENCE ===

def load_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[+] Model device: {device}")
    
    model = TransformerCausalDecoder(d_model=256, num_layers=4, d_audio=128, d_target=8)
    ckpt_path = ROOT / "models" / "checkpoints" / "transformer_phase12b_ep5.pt"
    
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, device

def temporal_nms(candidates, window_ms=100.0):
    if not candidates:
        return []
    sorted_cands = sorted(candidates, key=lambda x: x["prob"], reverse=True)
    keep = []
    while sorted_cands:
        best = sorted_cands.pop(0)
        keep.append(best)
        sorted_cands = [c for c in sorted_cands 
                       if not (c["type"] == best["type"] and abs(c["time"] - best["time"]) < window_ms)]
    return sorted(keep, key=lambda x: x["time"])

def generate_notes(audio_features, model, device, duration):
    """Run model inference for all 5 difficulties."""
    audio_features = audio_features.to(device)
    T_len = audio_features.shape[1]
    targets = torch.zeros((1, T_len, 8)).to(device)
    results = {}
    
    difficulties = ["Easy", "Normal", "Hard", "Expert", "Master"]
    density_factor = {"Easy": 0.25, "Normal": 0.5, "Hard": 0.75, "Expert": 1.0, "Master": 1.25}
    
    for diff_idx, diff_name in enumerate(difficulties):
        print(f"[+] Generating {diff_name}...")
        diff_tensor = torch.tensor([diff_idx]).to(device)
        
        with torch.no_grad():
            preds = model(audio_features, targets, diff_tensor)
        
        pres_prob = torch.sigmoid(preds["presence_logits"][0]).cpu().numpy()
        pos_pred = preds["position_pred"][0].cpu().numpy()
        
        candidates = []
        for t_idx in range(T_len):
            for h in range(2):
                if pres_prob[t_idx, h] > DETECTION_THRESHOLD:
                    candidates.append({
                        "time": float(t_idx * 0.1),
                        "type": int(h),
                        "prob": float(pres_prob[t_idx, h]),
                        "x": float(pos_pred[t_idx, h * 2] * 3.0),
                        "y": float(pos_pred[t_idx, h * 2 + 1] * 2.0)
                    })
        
        cleaned = temporal_nms(candidates, window_ms=NMS_WINDOW)
        
        # Apply density scaling
        target_count = int(len(cleaned) * density_factor[diff_name])
        if target_count < len(cleaned):
            cleaned = sorted(cleaned, key=lambda x: x["prob"], reverse=True)[:target_count]
            cleaned = sorted(cleaned, key=lambda x: x["time"])
        
        results[diff_name] = cleaned
        print(f"    {len(cleaned)} notes")
    
    return results

# === SYNTH CONVERSION ===

def seconds_to_tick(seconds: float, bpm: float) -> int:
    beats = seconds * bpm / 60.0
    return int(round(beats * INDEX_SCALE))

def tick_to_second(tick: int, bpm: float) -> float:
    beats = tick / INDEX_SCALE
    return beats * 60.0 / bpm

def tick_to_z(tick: int, bpm: float) -> float:
    return tick_to_second(tick, bpm) * TIME_SCALE

def build_beatmap_meta(notes_by_diff, bpm, song_name, artist, mapper,
                       audio_name="audio.ogg") -> dict:
    track = {}
    slides = {}
    lights = {}
    effects = {}
    
    for diff_name, notes in notes_by_diff.items():
        tick_notes = {}
        
        for n in notes:
            time_s = n["time"]
            tick = seconds_to_tick(time_s, bpm)
            tick_str = str(tick)
            if tick_str not in tick_notes:
                tick_notes[tick_str] = []
            
            x_w = n.get("x", 0)
            y_w = n.get("y", 0)
            z = tick_to_z(tick, bpm)
            hand_type = str(n.get("type", 0))
            
            note_obj = {
                "Time": tick,
                "Type": hand_type,
                "Position": [x_w, y_w, z],
                "Segments": None
            }
            tick_notes[tick_str].append(note_obj)
        
        track[diff_name] = tick_notes
        slides[diff_name] = []
        lights[diff_name] = []
        effects[diff_name] = []
    
    return {
        "Name": song_name,
        "Author": artist,
        "Beatmapper": mapper,
        "BPM": bpm,
        "AudioName": audio_name,
        "Artwork": "cover.jpg",
        "Track": track,
        "Slides": slides,
        "Lights": lights,
        "Effects": effects,
        "Bookmarks": {"BookmarksList": []}
    }

def package_synth(meta, audio_src, out_path):
    """Package as .synth ZIP with audio + cover."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        
        # Write meta
        with open(td_path / "beatmap.meta.bin", "w") as f:
            json.dump(meta, f)
        
        # Convert audio to OGG
        audio_src = Path(audio_src)
        if audio_src.suffix.lower() == ".ogg":
            shutil.copy2(audio_src, td_path / "audio.ogg")
        else:
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(audio_src), "-c:a", "libvorbis", "-q:a", "4",
                     str(td_path / "audio.ogg")],
                    check=True, capture_output=True
                )
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"[!] ffmpeg failed: {e}")
                print("    Falling back to copy as-is (game may not play it)")
                shutil.copy2(audio_src, td_path / f"audio{audio_src.suffix}")
        
        # Write a blank cover (or use a placeholder)
        (td_path / "cover.jpg").write_bytes(b"")  # Placeholder
        
        # Create ZIP
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in td_path.iterdir():
                zf.write(f, f.name)
    
    return True

def validate_synth(synth_path) -> dict:
    """Validate the generated .synth file."""
    synth_path = Path(synth_path)
    issues = []
    
    try:
        with zipfile.ZipFile(synth_path, "r") as zf:
            # Check required files
            files = zf.namelist()
            if "beatmap.meta.bin" not in files:
                issues.append("Missing beatmap.meta.bin")
            if not any("audio" in f for f in files):
                issues.append("Missing audio file")
            
            # Parse and validate
            with zf.open("beatmap.meta.bin") as f:
                meta = json.load(f)
            
            # Check structure
            if "Track" not in meta:
                issues.append("Missing Track field")
            if "BPM" not in meta:
                issues.append("Missing BPM")
            
            # Count notes per difficulty
            note_counts = {}
            for diff in ["Easy", "Normal", "Hard", "Expert", "Master"]:
                tracks = meta.get("Track", {}).get(diff, {})
                count = sum(len(notes) for notes in tracks.values()) if isinstance(tracks, dict) else 0
                if count == 0:
                    issues.append(f"{diff}: zero notes")
                note_counts[diff] = count
    
    except Exception as e:
        issues.append(f"Parse error: {e}")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "note_counts": note_counts if 'note_counts' in dir() else {}
    }

# === MAIN ===

def main():
    # Use existing track with pre-extracted features (bypasses iCloud issues with user MP3s)
    audio_path = "/Volumes/Second-Brain-1/AI/Synth/dataset/extracted/0007b2da6d9527ab/audio.ogg"
    feat_path = "/Volumes/Second-Brain-1/AI/Synth/dataset/audio_features/0007b2da6d9527ab.npz"
    output_dir = ROOT / "test_phase0"
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("  PHASE 0: End-to-End Pipeline Test")
    print("  Using existing track: 0007b2da6d9527ab")
    print("  (Pre-extracted features + existing game audio)")
    print("=" * 60)
    
    # 1. Load pre-extracted features
    print("\n=== STEP 1: Load Audio Features ===")
    try:
        data = np.load(feat_path)
        raw_mel = torch.from_numpy(data["audio_mel"]).float().unsqueeze(0)
        print(f"[+] Audio mel: shape={data['audio_mel'].shape}, uuid={data.get('uuid', 'unknown')}")
    except Exception as e:
        print(f"[✗] Feature loading failed: {e}")
        return 1
    
    # Compute augmented features using the PROVEN pipeline
    sys.path.insert(0, str(ROOT))
    from scripts.generate_gold_standard import compute_augmented_features, temporal_nms
    
    raw_mel = torch.from_numpy(data["audio_mel"]).float().unsqueeze(0)
    audio_features = compute_augmented_features(raw_mel)
    print(f"[✓] Audio features: {audio_features.shape}")
    
    # 2. Load model
    print("\n=== STEP 2: Model Loading ===")
    try:
        model, device = load_model()
        print(f"[✓] Model loaded on {device}")
    except Exception as e:
        print(f"[✗] Model loading failed: {e}")
        return 1
    
    # 3. Generate notes
    print("\n=== STEP 3: AI Generation ===")
    try:
        audio_features = audio_features.to(device)
        T_len = audio_features.shape[1]
        notes_by_diff = generate_notes(audio_features, model, device, T_len)
        total_notes = sum(len(n) for n in notes_by_diff.values())
        print(f"[✓] Total notes: {total_notes}")
    except Exception as e:
        print(f"[✗] Generation failed: {e}")
        return 1
    
    # 4. Convert to .synth
    print("\n=== STEP 4: .synth Conversion ===")
    try:
        bpm = float(data.get("bpm", 128.0))
        print(f"    BPM: {bpm}")
        
        meta = build_beatmap_meta(
            notes_by_diff, bpm,
            song_name="Phase0 Test Track",
            artist="Unknown",
            mapper="SynthGen AIv12b"
        )
        
        synth_path = output_dir / "phase0_test.synth"
        package_synth(meta, audio_path, synth_path)
        print(f"[✓] Saved to: {synth_path}")
    except Exception as e:
        print(f"[✗] Conversion failed: {e}")
        return 1
    
    # 5. Validate
    print("\n=== STEP 5: Validation ===")
    try:
        result = validate_synth(synth_path)
        if result["valid"]:
            print("[✓] Validation passed!")
            for diff, count in result["note_counts"].items():
                print(f"    {diff}: {count} notes")
        else:
            print("[!] Validation issues:")
            for issue in result["issues"]:
                print(f"    - {issue}")
    except Exception as e:
        print(f"[✗] Validation failed: {e}")
        return 1
    
    # 6. Summary
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"Output: {synth_path}")
    print(f"Size:   {synth_path.stat().st_size / 1024:.1f} KB")
    print(f"BPM:    {bpm}")
    for diff, notes in notes_by_diff.items():
        print(f"  {diff}: {len(notes)} notes")
    print()
    print("=== NEXT STEPS ===")
    print("1. Load the .synth file into SynthRiders")
    print(f"   File: {synth_path}")
    print("2. Check if it plays without errors")
    print("3. Report back: does it load? Does it crash? Does it feel playable?")
    print()

if __name__ == "__main__":
    sys.exit(main())
