# SynthRiders AI — Agent Log

## 2026-04-25 17:10 — Feature Engineering & Baseline Training Started

### What was done
1. Created `src/features/feature_engineering.py` — converts parsed JSON beatmaps to model tensors
   - Fixed-time frames (20ms)
   - 2D spatial grid (16x8 bins) + continuous positions
   - Handles all 4 note types (0,1,2,3 mapped to right/left)
2. Created `src/audio/audio_features.py` — audio extractor + synthetic generator
3. Created `src/models/baseline.py` — Conv1D encoder + LSTM decoder
   - 1.1M parameters
   - Multi-task: presence (BCE), position (MSE), rail (BCE)
4. Created `src/training/train_baseline.py` — training loop with metrics
5. Created `scripts/visualize_predictions.py` — visualization tool
6. Generated 1,146 feature files from parsed beatmaps (Hard difficulty only)

### Issues encountered
- **MPS crash**: `validateComputeFunctionArguments:1066` — PyTorch MPS bug with division ops
  - **Fix**: Switched to CPU training for overfit test
  - **Status**: Training running on CPU (slower but stable)

### Blockers
- CPU training is slow (~2+ min per epoch estimated)
- Need to wait for overfit test completion before scaling

### Next steps
1. Wait for baseline overfit test to complete (100 maps, 100 epochs)
2. Generate loss curves and visualizations
3. If recall > 90% and timing < 50ms: proceed to full-corpus training
4. If not: debug model architecture / data pipeline

### Key decisions made
- Frame resolution: **20ms fixed-time** (not beat-synchronous)
- Spatial encoding: **Both grid + continuous** (grid for occupancy, continuous for positions)
- Rail encoding: **Binary mask only** for baseline (full path encoding in v2)
- Difficulty: Training on **Hard** first (1,146 maps available)
