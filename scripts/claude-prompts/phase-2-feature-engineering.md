# SynthRiders AI — Phase 2: Feature Engineering & Baseline Model
## Autonomous Agent Prompt
**Date:** 2026-04-25
**Project root:** `/Volumes/Second-Brain-1/AI/Synth/`
**Agent:** Claude Code (via `claude --acp --stdio`)

---

## 1. What Just Happened (State of Play)

Corpus ingestion is **COMPLETE**. Summary:

| Metric | Value |
|--------|-------|
| .synth files found | 8,301 |
| Decrypted | 8,300 (99.99%) |
| Parsed beatmaps | 4,638 |
| Total notes | 5,202,229 |
| Avg notes/map | 1,122 |

**Difficulty distribution:** Master 44.7% (2,075 maps), Expert 37.8% (1,751), Hard 24.8% (1,152).

**The .synth format** is a ZIP archive containing `beatmap.meta.bin`, which despite the extension is **UTF-8 JSON with BOM**. It stores an entire beatmap — metadata, all difficulties, and base64-encoded artwork — in a single file. Only ~55.9% of extracted archives had this file; the rest are legacy/incomplete.

**Deliverables already created:**
- `dataset/extracted/<uuid>/` — decrypted ZIP contents
- `dataset/parsed/<uuid>.json` — structured beatmap data
- `dataset/parsed/index.json` — master parsed index
- `dataset/reports/ingestion-report-2026-04-25.md` — full report
- `scripts/batch_extract.py` — batch decryptor
- `scripts/parse_beatmap_data.py` — beatmap-to-JSON parser
- `dataset/specs/beatmap.data.md` — format documentation

---

## 2. Parsed Data Schema (CRITICAL — understand this)

Each `<uuid>.json` has this top-level structure:

```json
{
  "status": "success",
  "path": ".../beatmap.meta.bin",
  "metadata": {
    "name": "Supernova",
    "author": "Within Temptation",
    "bpm": 127.0,
    "offset": 0.0,
    "beatmapper": "Technical",
    "editor_version": "1.9.7-10",
    "production_mode": true,
    "modified_time": 1607372250
  },
  "difficulties": {
    "Normal": { "note_count": 1016, "notes": [...] }
  },
  "stats": {
    "total_notes": 1016,
    "difficulty_count": 1,
    "has_base64_artwork": true,
    "file_size_kb": 712.9
  }
}
```

**Note structure** (each element in `difficulties.<name>.notes`):

| Field | Type | Description |
|-------|------|-------------|
| `time` | float | Timestamp in **milliseconds** |
| `x` | float | Horizontal position (~ -0.54 to +0.54) |
| `y` | float | Vertical position (~ -0.41 to +0.41) |
| `z` | float | Depth / timing axis. **NOT world Z — this is a derived playback coordinate.** Time is the true temporal axis. |
| `type` | int | **0 = Right hand, 1 = Left hand** |
| `direction` | int | Always 0 in the corpus seen so far. Might indicate rail entry angle in newer maps. |
| `combo_id` | int | -1 for most notes. Combo grouping ID for multi-hit patterns. |
| `id` | string | Unique note ID, includes timestamp and handedness |
| `segments` | list[list[float]]\|null | **Rails / slides.** If present, it's a list of [x, y, z] points the hand follows after hitting this note. A rail note IS both a hit point AND a path. |

**Key observations:**
- `type`: 0 = Right, 1 = Left. SynthRiders uses left/right hand assignment explicitly.
- `segments`: Present on ~25-30% of notes. These are rails/slides — critical for flow and difficulty prediction.
- The `z` coordinate is **not independent time**; it's a playback parameter. Use `time` as the canonical temporal axis.
- x/y coordinates cluster around grid positions (looks like a quantized grid with ~6-8 horizontal slots and ~4-5 vertical slots).

Verify with: `python3 -c "import json; data=json.load(open('/Volumes/Second-Brain-1/AI/Synth/dataset/parsed/001c1ec9e8d98b55.json')); print(len(data['difficulties']['Normal']['notes'])); print(data['difficulties']['Normal']['notes'][0])"`

---

## 3. Your Mission: Feature Engineering & Baseline Model

The goal is to build a **trainable ML pipeline** that can predict beatmaps from audio. This phase has three tracks — complete them in order:

### Track A: Feature Engineering (MUST DO FIRST)

Create a module that converts each parsed beatmap into model-ready tensors. The output should be a **structured dataset** on disk (e.g., Parquet, NumPy .npz, or PyTorch .pt) that can be loaded efficiently during training.

For each beatmap, extract:

1. **Temporal grid:** Discretize the song into fixed-width bins (e.g., 10ms or beat-synchronous). Output: a sequence of frames.
2. **Note occupancy tensor:** Per-frame, per-hand (R/L), a 2D spatial grid or continuous (x,y) representation of note presence.
3. **Rail tensor:** Per-frame, a binary or path-encoded feature indicating whether a rail is active and its trajectory.
4. **Beat-aligned metadata:** BPM, time signature (if inferable), downbeat positions, beat phase.
5. **Difficulty label:** Normal / Hard / Expert / Master (categorical).

**Important design decisions you must make and document:**
- **Frame resolution:** Fixed-time (e.g., 10ms, 20ms) vs. beat-synchronous (variable length per song). Beat-synchronous is musically meaningful but harder to batch. Recommend starting with fixed-time and providing a beat-synchronous dataloader as an option.
- **Spatial encoding:** Continuous (x,y floats) vs. discretized grid. SynthRiders looks like it uses a latent grid. Try both; a VQVAE or learned embedding might be better than raw coordinates.
- **Rail encoding:** Segments are polyline paths. Options: (a) rasterize into per-frame occupancy, (b) parametrize as Bézier or spline control points, (c) treat as additional note positions in subsequent frames.

Create:
- `scripts/feature_engineering.py` or `src/features/` package
- `dataset/features/` directory for precomputed features (keep it organized by UUID)
- A `SynthBeatmapDataset` PyTorch Dataset class that can load from these features

### Track B: Audio Analysis Pipeline (CAN RUN IN PARALLEL WITH A)

The model needs **audio features** as input. We don't have the raw audio files (they're not in the .synth archives), but we MUST design the pipeline that will consume them.

Create:
- `src/audio/` package with feature extractors using **librosa** (or torchaudio):
  - Mel spectrogram (80-128 bins, standard for music ML)
  - Chroma (12-bin pitch class) for harmonic structure
  - Tempogram / onset envelope for rhythmic events
  - Beat-synchronous aggregation
  - Constant-Q transform (CQT) as an alternative to STFT
- `scripts/extract_audio_features.py` — a standalone script that takes an audio file path and outputs a feature tensor file. This is our inference-time entry point.
- Document the expected audio format (WAV, 44.1kHz or 22.05kHz, mono) and feature shapes in `dataset/specs/audio-features.md`.

**Without the audio files, you can't train end-to-end yet.** But you CAN:
1. Design the audio encoder architecture (e.g., 1D CNN on raw waveform, or 2D CNN on spectrogram).
2. Build the dataloader to accept pre-extracted audio features from disk.
3. Create a synthetic audio feature generator for the baseline overfit test (white noise → random spectrogram) so we can validate the *map decoder* independently.

### Track C: Baseline Model Architecture & Overfit Test

Build a minimal, **overfittable** model. The goal isn't state-of-the-art — it's proving the data pipeline works end-to-end.

**Recommended architecture** (start simple):

```
Audio Features (T, F) 
  -> Audio Encoder (Conv1D stack or small Transformer) 
  -> Latent sequence (T, D)
  -> Note Decoder (LSTM or Transformer with causal masking) 
  -> Per-frame note predictions:
       - Note presence (binary or multi-class per hand)
       - X, Y coordinates (regression)
       - Rail flag + path params
```

**Train on a tiny subset first:**
- Pick 100 maps (or even 10) from the corpus.
- Use synthetic/random audio features for now (since we don't have matching audio).
- Train until training loss drops to near-zero (overfit).
- **Success criterion:** The model can reconstruct the training beatmaps from synthetic audio input with >90% note recall and <50ms timing error.

This proves: (a) the feature encoder works, (b) the decoder can express beatmap structure, (c) the data pipeline has no bugs.

Create:
- `src/models/` package
- `src/training/train_baseline.py` — training loop
- `scripts/overfit_baseline.sh` — one-command run
- Save model checkpoints to `models/checkpoints/`

---

## 4. Infrastructure Decision

There is an **existing project** at `/Volumes/Second-Brain-1/Home/Desktop/SynthGen/` with a FastAPI backend (4,413 LOC) + React frontend. It was trained on **synthetic data only** and achieved 57% accuracy on a toy test. 

**Your choice:** Salvage or start fresh.
- **Salvage:** Read the existing FastAPI backend to understand their data model, API design, and ML module. Adapt their code if it's clean.
- **Start fresh:** If the existing code is a mess (synthetic-only pipeline, bad architecture, hardcoded shapes), create a new `src/` layout under `/Volumes/Second-Brain-1/AI/Synth/`.

**My recommendation:** Start fresh for the ML core, but read the existing project first to steal good ideas. The existing frontend can be adapted later for demonstration.

---

## 5. Deliverables Checklist

Complete these and report back:

- [ ] Feature engineering module converting parsed JSON → model tensors (Track A)
- [ ] Audio feature extractor design + standalone script (Track B)
- [ ] Baseline model architecture (Track C)
- [ ] Overfit test on 10-100 maps with synthetic audio input
- [ ] Training loss curves and sample predictions (visualized — e.g., plot note positions over time)
- [ ] Updated project documentation (README.md, any new spec docs)
- [ ] `requirements.txt` or `pyproject.toml` with all dependencies
- [ ] One-command training script (`make train-baseline` or `python scripts/train_baseline.py`)

---

## 6. Constraints & Notes

- **Hardware:** Mac mini M4 Pro, 64GB RAM. Training must run on Apple Silicon (MLX or PyTorch MPS). No CUDA available.
- **Storage:** ~2TB external, but keep intermediate files tidy. Parquet or compressed NPZ preferred over raw JSON for features.
- **Python:** 3.14.3 via Homebrew (`/opt/homebrew/bin/python3`)
- **No audio files yet:** We don't have the original MP3s/WAVs matching the beatmaps. Design for this; generate synthetic audio features for the baseline test.
- **Keep it simple:** This is a research spike, not production code. Optimize for experimentation speed, not deployment.

---

## 7. How to Report Progress

Update the following file with a brief status log after each significant milestone:

`/Volumes/Second-Brain-1/AI/Synth/AGENT_LOG.md`

Format:
```markdown
## YYYY-MM-DD HH:MM — [Milestone Name]
- What was done
- Key decisions made
- Blockers (if any)
- Next step
```

When you hit a decision that affects architecture, pause and write the options + your recommendation into `AGENT_LOG.md` before proceeding.

---

**GOAL:** End this phase with a working training loop, a validated data pipeline, and a model that can overfit 100 beatmaps from synthetic audio input. The next phase will scale to the full corpus and add real audio features.

**Start by reading the existing codebase, the parsed data, and the existing SynthGen project — then build.**
