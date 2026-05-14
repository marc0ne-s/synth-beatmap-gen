"""
SynthGen Backend API
FastAPI server wrapping the AI pipeline and .synth export layer.
Runs as a separate Python process; frontend calls via localhost.
"""

import os
import sys
import shutil
import json
import zipfile
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import torch
import librosa
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add project root to path
ROOT = Path("/Volumes/Second-Brain-1/AI/Synth")
sys.path.insert(0, str(ROOT))

# === CONSTANTS ===
TIME_SCALE = 20.0
INDEX_SCALE = 64
GRID_SCALE = 0.1365
X_OFFSET = 0.002
Y_OFFSET = 0.0012

# === PYDANTIC MODELS ===

class GenerateRequest(BaseModel):
    audio_path: str
    bpm: Optional[float] = None
    difficulties: List[str] = ["Easy", "Normal", "Hard", "Expert", "Master"]
    nms_window: float = 100.0
    detection_threshold: float = 0.5

class NoteOut(BaseModel):
    time: float
    x: float
    y: float
    type: int
    prob: float

class DifficultyOut(BaseModel):
    notes: List[NoteOut]
    rails: List[dict] = []
    slides: List[dict] = []

class GenerateResponse(BaseModel):
    success: bool
    uuid: str
    duration: float
    bpm: float
    difficulties: Dict[str, DifficultyOut]
    version_tag: str = "AIv12b.100ms"

class ConvertRequest(BaseModel):
    project_path: str
    audio_path: str
    cover_path: Optional[str] = None
    out_path: str
    song_name: Optional[str] = None
    artist: Optional[str] = None
    mapper: str = "SynthGen AI"
    bpm: Optional[float] = None

class ConvertResponse(BaseModel):
    success: bool
    synth_path: str
    note_counts: Dict[str, int]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    version: str = "0.1.0"

# === SYNTH FORMAT CONVERTER ===

def seconds_to_tick(seconds: float, bpm: float) -> int:
    beats = seconds * bpm / 60.0
    return int(round(beats * INDEX_SCALE))

def tick_to_second(tick: int, bpm: float) -> float:
    beats = tick / INDEX_SCALE
    return beats * 60.0 / bpm

def tick_to_z(tick: int, bpm: float) -> float:
    seconds = tick_to_second(tick, bpm)
    return seconds * TIME_SCALE

def world_to_grid(x: float, y: float) -> List[float]:
    """Convert world-space (-3..3 x, -2..2 y) to grid units."""
    return [x / GRID_SCALE + X_OFFSET, y / GRID_SCALE + Y_OFFSET]

def grid_to_world(gx: float, gy: float) -> List[float]:
    """Grid units to world-space."""
    return [(gx - X_OFFSET) * GRID_SCALE, (gy - Y_OFFSET) * GRID_SCALE]

def build_beatmap_meta(notes_by_diff: Dict[str, List[Dict]], bpm: float,
                       song_name: str, artist: str, mapper: str,
                       audio_name: str, cover_name: str) -> Dict:
    """Build the full beatmap.meta.bin JSON structure."""
    track = {}
    slides = {}
    lights = {}
    effects = {}
    
    for diff_name, notes in notes_by_diff.items():
        # Convert notes to tick-based format
        tick_notes: Dict[str, List[dict]] = {}
        tick_walls: List[dict] = []
        
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
            
            # Build note object
            note_obj = {
                "Time": tick,
                "Type": hand_type,
                "Position": [x_w, y_w, z],
                "Segments": None
            }
            
            # Rail detection: sustained same-hand notes within 200ms
            if n.get("is_rail"):
                note_obj["Segments"] = [
                    {"Position": [x_w, y_w, z]}
                ]
            
            tick_notes[tick_str].append(note_obj)
        
        track[diff_name] = tick_notes
        slides[diff_name] = tick_walls
        lights[diff_name] = []  # TODO: add basic lighting
        effects[diff_name] = []
    
    meta = {
        "Name": song_name,
        "Author": artist,
        "Beatmapper": mapper,
        "BPM": bpm,
        "AudioName": audio_name,
        "Artwork": cover_name if cover_name else "cover.jpg",
        "Track": track,
        "Slides": slides,
        "Lights": lights,
        "Effects": effects,
        "Bookmarks": {"BookmarksList": []}
    }
    return meta

def heuristic_rails(notes: List[Dict], time_tol_ms: float = 200.0, 
                    pos_tol: float = 0.5) -> List[Dict]:
    """Heuristic: detect sustained sequences as rails."""
    if len(notes) < 2:
        return []
    
    # Sort by time
    notes_sorted = sorted(notes, key=lambda n: n["time"])
    rails = []
    rail_id = 0
    i = 0
    
    while i < len(notes_sorted) - 1:
        # Find sequences of same-type notes close together
        seq = [notes_sorted[i]]
        j = i + 1
        while j < len(notes_sorted):
            dt = (notes_sorted[j]["time"] - notes_sorted[-1]["time"]) * 1000
            if dt > time_tol_ms:
                break
            # Check hand type match and position continuity
            if notes_sorted[j].get("type") != seq[0].get("type"):
                break
            dx = abs(notes_sorted[j].get("x", 0) - seq[-1].get("x", 0))
            dy = abs(notes_sorted[j].get("y", 0) - seq[-1].get("y", 0))
            if dx > pos_tol or dy > pos_tol:
                break
            seq.append(notes_sorted[j])
            j += 1
        
        if len(seq) >= 3:
            # Mark all notes in sequence as rails (except first/last if needed)
            for n in seq:
                n["is_rail"] = True
                n["rail_id"] = f"rail_{rail_id}"
            rails.append(seq)
            rail_id += 1
            i = j
        else:
            i += 1
    
    return rails

def package_synth(meta: Dict, audio_src: Path, cover_src: Optional[Path], 
                  out_path: Path) -> bool:
    """Package beatmap.meta.bin + audio + cover as .synth ZIP."""
    try:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            
            # Write beatmap.meta.bin
            meta_path = td_path / "beatmap.meta.bin"
            with open(meta_path, "w") as f:
                json.dump(meta, f)
            
            # Copy audio (convert to OGG if needed)
            audio_ext = audio_src.suffix.lower()
            if audio_ext == ".ogg":
                shutil.copy2(audio_src, td_path / "audio.ogg")
                audio_name = "audio.ogg"
            else:
                # Try to convert to OGG using ffmpeg
                try:
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", str(audio_src), "-c:a", "libvorbis", "-q:a", "4",
                         str(td_path / "audio.ogg")],
                        check=True, capture_output=True
                    )
                    audio_name = "audio.ogg"
                except (subprocess.CalledProcessError, FileNotFoundError):
                    # Fallback: copy as-is
                    dest_name = f"audio{audio_ext}"
                    shutil.copy2(audio_src, td_path / dest_name)
                    audio_name = dest_name
            
            # Copy cover
            if cover_src and cover_src.exists():
                cover_name = f"cover{cover_src.suffix}"
                shutil.copy2(cover_src, td_path / cover_name)
            else:
                cover_name = None
            
            # Update meta with correct audio/cover names
            meta["AudioName"] = audio_name
            if cover_name:
                meta["Artwork"] = cover_name
            
            # Re-write meta with updated names
            with open(meta_path, "w") as f:
                json.dump(meta, f)
            
            # Create ZIP
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(meta_path, "beatmap.meta.bin")
                zf.write(td_path / audio_name, audio_name)
                if cover_name:
                    zf.write(td_path / cover_name, cover_name)
            
            return True
    except Exception as e:
        print(f"[ERROR] Package failed: {e}")
        return False

# === AI INFERENCE ===

def load_model(checkpoint_name: str = "transformer_phase12b_ep5.pt"):
    """Lazy-load the transformer model."""
    from src.models.transformer import TransformerCausalDecoder
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = TransformerCausalDecoder(d_model=256, num_layers=4, d_audio=128, d_target=8)
    
    ckpt_path = ROOT / "models" / "checkpoints" / checkpoint_name
    if not ckpt_path.exists():
        raise RuntimeError(f"Checkpoint not found: {ckpt_path}")
    
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    return model, device

_model_cache = {}

def get_model():
    if "model" not in _model_cache:
        _model_cache["model"], _model_cache["device"] = load_model()
    return _model_cache["model"], _model_cache["device"]

def compute_augmented_features(mel: torch.Tensor) -> torch.Tensor:
    """Compute spectral features from mel spectrogram."""
    T, num_freq = mel.shape[1], mel.shape[2]
    mel_sq = mel[0]
    shifted_mel = torch.cat([torch.zeros(1, num_freq, device=mel.device), mel_sq[:-1, :]], dim=0)
    delta = mel_sq - shifted_mel
    delta_cpu = delta.cpu().unsqueeze(0)
    delta_44 = torch.nn.functional.adaptive_avg_pool1d(delta_cpu, 44).squeeze(0).to(mel.device)
    flux = torch.nn.functional.relu(delta).sum(dim=1, keepdim=True) / 10.0
    f_idx = torch.arange(num_freq, dtype=torch.float32, device=mel.device) / (num_freq - 1)
    mel_sum = mel_sq.sum(dim=1, keepdim=True) + 1e-8
    centroid = (mel_sq * f_idx.unsqueeze(0)).sum(dim=1, keepdim=True) / mel_sum
    f_diff_sq = (f_idx.unsqueeze(0) - centroid) ** 2
    bandwidth = torch.sqrt((mel_sq * f_diff_sq).sum(dim=1, keepdim=True) / mel_sum)
    rms = torch.sqrt((mel_sq ** 2).mean(dim=1, keepdim=True))
    augmented = torch.cat([mel_sq, delta_44, flux, centroid, bandwidth, rms], dim=1)
    return augmented.unsqueeze(0)

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

def extract_audio_features(audio_path: str) -> tuple:
    """Extract mel spectrogram + metadata from audio file."""
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    
    # Compute BPM if not provided
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo) if isinstance(tempo, (int, float, np.number)) else 128.0
    
    # Mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Duration
    duration = librosa.get_duration(y=y, sr=sr)
    
    return mel_db, sr, bpm, duration, y

def infer_map(audio_path: str, bpm_hint: Optional[float] = None,
              difficulties: List[str] = ["Easy", "Normal", "Hard", "Expert", "Master"],
              nms_window: float = 100.0, threshold: float = 0.5) -> Dict:
    """Run the full AI pipeline on an audio file."""
    model, device = get_model()
    
    # Extract features
    mel_db, sr, bpm_est, duration, raw_audio = extract_audio_features(audio_path)
    bpm = bpm_hint if bpm_hint else bpm_est
    
    # Convert mel to tensor
    mel_tensor = torch.from_numpy(mel_db).float().transpose(0, 1).unsqueeze(0).to(device)
    
    # Compute augmented features matching the model's expected input
    # Resample to ~100ms frames
    hop = int(sr * 0.1)  # ~100ms per frame
    frames = []
    for i in range(0, len(raw_audio) - hop, hop):
        frame = raw_audio[i:i + hop]
        frame_mel = librosa.feature.melspectrogram(y=frame, sr=sr, n_mels=128, n_fft=2048, hop_length=hop)
        frame_db = librosa.power_to_db(frame_mel, ref=np.max)
        frame_tensor = torch.from_numpy(frame_db).float().mean(dim=1)
        frames.append(frame_tensor)
    
    if not frames:
        raise RuntimeError("Audio too short for feature extraction")
    
    mel_per_frame = torch.stack(frames).unsqueeze(0).to(device)
    audio = compute_augmented_features(mel_per_frame)
    T_len = audio.shape[1]
    
    targets = torch.zeros((1, T_len, 8)).to(device)
    results = {}
    diff_map = {"Easy": 0, "Normal": 1, "Hard": 2, "Expert": 3, "Master": 4}
    
    for diff_name in difficulties:
        diff_idx = diff_map.get(diff_name, 2)
        diff_tensor = torch.tensor([diff_idx]).to(device)
        
        with torch.no_grad():
            preds = model(audio, targets, diff_tensor)
        
        pres_prob = torch.sigmoid(preds["presence_logits"][0]).cpu().numpy()
        pos_pred = preds["position_pred"][0].cpu().numpy()
        
        candidates = []
        for t_idx in range(T_len):
            for h in range(2):
                if pres_prob[t_idx, h] > threshold:
                    candidates.append({
                        "time": float(t_idx * 0.1),
                        "type": int(h),
                        "prob": float(pres_prob[t_idx, h]),
                        "x": float(pos_pred[t_idx, h * 2] * 3.0),
                        "y": float(pos_pred[t_idx, h * 2 + 1] * 2.0)
                    })
        
        cleaned = temporal_nms(candidates, window_ms=nms_window)
        
        # Apply density scaling per difficulty
        density_factor = {"Easy": 0.25, "Normal": 0.5, "Hard": 0.75, "Expert": 1.0, "Master": 1.25}
        target_count = int(len(cleaned) * density_factor.get(diff_name, 1.0))
        if target_count < len(cleaned):
            cleaned = sorted(cleaned, key=lambda x: x["prob"], reverse=True)[:target_count]
            cleaned = sorted(cleaned, key=lambda x: x["time"])
        
        # Detect rails
        heuristic_rails(cleaned)
        
        results[diff_name] = {
            "notes": cleaned,
            "rails": [],
            "slides": []
        }
    
    return {
        "success": True,
        "uuid": Path(audio_path).stem,
        "duration": duration,
        "bpm": bpm,
        "difficulties": results,
        "version_tag": "AIv12b.100ms"
    }

# === FASTAPI APP ===

app = FastAPI(title="SynthGen API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "tauri://localhost"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health():
    try:
        model, device = get_model()
        return HealthResponse(status="ok", model_loaded=True, device=str(device))
    except Exception as e:
        return HealthResponse(status=f"model_error: {e}", model_loaded=False, device="cpu")

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    try:
        audio_path = Path(req.audio_path)
        if not audio_path.exists():
            raise HTTPException(status_code=400, detail=f"Audio file not found: {req.audio_path}")
        
        result = infer_map(
            str(audio_path),
            bpm_hint=req.bpm,
            difficulties=req.difficulties,
            nms_window=req.nms_window,
            threshold=req.detection_threshold
        )
        return GenerateResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/convert", response_model=ConvertResponse)
async def convert(req: ConvertRequest):
    try:
        project_path = Path(req.project_path)
        if not project_path.exists():
            raise HTTPException(status_code=400, detail=f"Project not found: {req.project_path}")
        
        with open(project_path) as f:
            project = json.load(f)
        
        audio_src = Path(req.audio_path)
        if not audio_src.exists():
            raise HTTPException(status_code=400, detail=f"Audio file not found: {req.audio_path}")
        
        cover_src = Path(req.cover_path) if req.cover_path else None
        out_path = Path(req.out_path)
        
        bpm = req.bpm or project.get("bpm", 128.0)
        song_name = req.song_name or project.get("title", "Untitled")
        artist = req.artist or project.get("artist", "Unknown")
        
        # Build notes_by_diff
        notes_by_diff = {}
        note_counts = {}
        for diff_name in project.get("difficulties", {}):
            diff_data = project["difficulties"][diff_name]
            notes_by_diff[diff_name] = diff_data.get("notes", [])
            note_counts[diff_name] = len(notes_by_diff[diff_name])
        
        # Build beatmap meta
        meta = build_beatmap_meta(
            notes_by_diff, bpm, song_name, artist, req.mapper,
            audio_src.name, cover_src.name if cover_src else None
        )
        
        # Package as .synth
        success = package_synth(meta, audio_src, cover_src, out_path)
        
        return ConvertResponse(
            success=success,
            synth_path=str(out_path),
            note_counts=note_counts
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "SynthGen API", "version": "0.1.0"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")
