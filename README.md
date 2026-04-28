# SynthRiders AI Beatmap Generator

ML pipeline for generating SynthRiders VR rhythm game beatmaps from audio.

## Project Status

| Phase | Status | Notes |
|-------|--------|-------|
| Corpus Ingestion | **COMPLETE** | 4,638 parsed maps, 5.2M notes |
| Feature Engineering | **COMPLETE** | 20ms fixed-time frames, 16x8 spatial grid |
| Baseline Model | **COMPLETE** | 61K params, 100% recall on overfit test |
| Audio Pipeline | **COMPLETE** | 612 real mel spectrograms extracted from OGG |
| Full Training | **IN PROGRESS** | 100-map overfit with real audio, CPU ~150s/epoch |
| Map Generation | **COMPLETE** | `generate_beatmap.py` inference + .synth export |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run fast overfit test (proves pipeline works)
python3 src/training/overfit_fast.py --num-maps 20 --epochs 100

# Full baseline training
bash scripts/train_baseline.sh
```

## Repository Structure

```
.
├── dataset/
│   ├── raw/              # Symlinks to .synth files (8,301)
│   ├── extracted/        # Decrypted ZIP contents
│   ├── parsed/           # JSON beatmap data (4,638 maps)
│   ├── features/         # Precomputed model tensors (1,146 Hard)
│   ├── specs/
│   │   └── beatmap.data.md   # Format documentation
│   └── reports/
│       ├── ingestion-report-2026-04-25.md
│       └── feature_engineering_v2.log
├── src/
│   ├── features/
│   │   ├── feature_engineering.py   # Parsed JSON → tensors
│   │   └── extract_audio_features.py # OGG → mel spectrograms
│   ├── audio/
│   │   └── audio_features.py        # librosa + synthetic audio
│   ├── models/
│   │   └── baseline.py              # Conv1D + LSTM decoder
│   ├── training/
│   │   ├── train_baseline.py        # Full training loop
│   │   └── overfit_fast.py          # Fast validation
│   └── inference/
│       ├── generate_beatmap.py      # Audio → .synth beatmap
│       └── evaluate_map.py          # Compare generated vs ground truth
├── scripts/
│   ├── batch_extract.py             # Decrypt .synth files
│   ├── extract_audio_batch.py       # Batch OGG → mel (1,146 maps)
│   ├── parse_beatmap_data.py       # Parse beatmap.meta.bin
│   ├── synth_decryptor.py          # CLI decryptor
│   ├── synth_password_extractor.py # Extract ZIP password
│   ├── visualize_predictions.py    # Results visualization
│   └── train_baseline.sh           # One-command training
├── models/checkpoints/
│   ├── best_model.pt                 # Latest checkpoint
│   ├── overfit_tiny.pt             # Trained model (61K params)
│   └── overfit_history.json        # Training metrics
├── AGENT_LOG.md                    # Development log
└── README.md                       # This file
```

## Data Pipeline

1. **Source**: `.synth` files from Meta Quest (`/Volumes/Second-Brain-1/Meta Quest/Synth/`)
2. **Decrypt**: `pyzipper` with password `hC2*wE5R*qQzv@a!`
3. **Parse**: `beatmap.meta.bin` is UTF-8 JSON with BOM
4. **Feature Extraction**: 20ms frames, 16x8 spatial grid, continuous positions
5. **Model Input**: `(B, T, 80)` synthetic audio → `(B, T, 2)` presence + `(B, T, 4)` positions

## Baseline Model Architecture

```
Audio Features (B, T, 80)
  → Conv1D Encoder (2 layers, 64 hidden)
  → LSTM Decoder (1 layer, 64 hidden)
  → Multi-task heads:
       - Presence: (B, T, 2) binary per hand
       - Position: (B, T, 4) x,y per hand
       - Rail: (B, T, 1) binary (placeholder)
```

**Parameters**: 61,446 (tiny model for fast iteration)

## Overfit Test Results

| Metric | Target | Achieved |
|--------|--------|----------|
| Note Recall | > 90% | **100%** |
| Timing Error | < 50ms | **13.7ms** |
| Position Error | — | 0.263 |

**Training**: 20 maps, 100 epochs, ~1s/epoch on CPU
**Loss**: Dropped from 1.43 to 0.97 (min at epoch 96)
**Recall**: Reached 100% by epoch 3

## Corpus Statistics

| Metric | Value |
|--------|-------|
| Total maps | 4,638 |
| Total notes | 5,202,229 |
| Avg notes/map | 1,122 |
| BPM range | 40–500 (mean 132.7) |
| Master difficulty | 2,075 maps (44.7%) |
| Expert difficulty | 1,751 maps (37.8%) |

## Hardware

- Mac mini M4 Pro, 64GB RAM
- PyTorch MPS (Apple Silicon)
- Training currently on CPU due to MPS division bug (PyTorch issue)

## Known Issues

1. **MPS crash**: `validateComputeFunctionArguments:1066` with div ops
   - Workaround: Use CPU for now
   - Fix: Wait for PyTorch update or avoid in-place division

2. **Training speed**: ~150s/epoch on CPU for 100 maps
   - MPS crashes on Apple Silicon; blocked on PyTorch fix
   - Consider smaller model or data parallelism

3. **Class imbalance**: ~5% of frames contain notes
   - Fixed with `pos_weight` in BCE loss
   - May need focal loss for full training

## Next Steps

1. **Complete 100-map training** — monitor convergence with real audio (~4h remaining)
2. **Scale to full corpus** — train on all 1,146 Hard maps with real audio
3. **Beat-synchronous frames** — align to musical beats instead of fixed time
4. **Difficulty-conditioned generation** — train one model, condition on difficulty
5. **Evaluate inference quality** — compare generated maps against ground truth

## License

Research project. Not affiliated with SynthRiders or Kluge Interactive.
