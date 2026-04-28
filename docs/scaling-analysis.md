# SynthRiders ML Pipeline — Scaling Analysis
## From 100 Maps to 500+ on M4 Pro 64GB

**Date:** 2026-04-25
**Hardware:** Mac Mini M4 Pro, 64GB unified memory, Apple Silicon (MPS backend)
**Current state:** 100-map overfit test running on CPU (MPS not yet engaged)

---

## 1. Current Bottleneck Assessment

### 1.1 Training Bottleneck: CPU-bound, not memory-bound

Current metrics from training log:
- **~146s/epoch** on CPU
- **Batch size:** 4
- **100 maps × ~15,000 frames avg = ~1.5M frames total**
- **Model:** ~1.2M parameters (Conv1D encoder + LSTM decoder)

**Why CPU?** There is no `torch.backends.mps.is_available()` guard failure — the script falls back to CPU if MPS is unavailable. Verify with:

```bash
python3 -c "import torch; print(torch.backends.mps.is_available()); print(torch.backends.mps.is_built())"
```

If this returns `(True, True)`, the model should already be on MPS. If training is still slow, the bottleneck is likely:

1. **Data loading** — reading NPZ feature files from disk each epoch
2. **Batch collation** — padding variable-length sequences on CPU before sending to MPS
3. **Small batch underutilization** — MPS works best with larger batches (16–32)

### 1.2 Memory Footprint Estimate

Per training sample (at max length):
- Audio features: 15,000 frames × 80 mels × 4 bytes = **4.8 MB**
- Note occupancy: 15,000 × 2 × 16 × 8 × 4 bytes = **15.4 MB**
- Note positions: 15,000 × 2 × 2 × 4 bytes = **240 KB**
- Note presence: 15,000 × 2 × 4 bytes = **120 KB**
- **Total per sample: ~20.5 MB**

Batch size 4: **~82 MB** per batch

Model parameters (~1.2M params × 4 bytes) + gradients + optimizer state (Adam = 2× params):
- **~14 MB** for model + optimizer

**Total training memory per batch: ~100 MB**

With 64GB unified memory, we can easily fit **batch size 32 (~800 MB)** or even **64 (~1.6 GB)**. Memory is NOT the bottleneck.

---

## 2. Scaling Recommendations

### 2.1 Immediate Wins (before scaling to 500)

| Fix | Expected Gain | Effort |
|-----|-------------|--------|
| **Verify MPS is actually used** | 5–10× speedup | 1 min |
| **Increase batch size to 16–32** | 2–4× throughput | 5 min |
| **Preload entire dataset to RAM** | 1.5× (eliminates disk I/O) | 20 min |
| **Pin memory for MPS** | 1.2× | 5 min |
| **Use persistent workers** (`num_workers > 0`) | 1.3× | 5 min |

**Combined potential: 10–30× faster epochs**, dropping from 146s to **5–15s/epoch**.

#### 2.1.1 Code Changes for MPS + Larger Batch

In `train_baseline.py`:

```python
# 1. Ensure MPS is actually used
if torch.backends.mps.is_available():
    device = torch.device("mps")
    # MPS benefits from larger batches
    batch_size = 16  # or 32
else:
    device = torch.device("cpu")
    batch_size = 4

# 2. Pre-load dataset into RAM
def preload_dataset(dataset):
    """Pre-load all samples into CPU RAM before training."""
    print("[+] Pre-loading dataset into RAM...")
    cached = []
    for i in range(len(dataset)):
        cached.append(dataset[i])
    return cached

# Use a simple InMemoryDataset wrapper
class InMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

# In train():
dataset = SynthBeatmapDataset(...)
cached_samples = preload_dataset(dataset)
dataset = InMemoryDataset(cached_samples)

# 3. Use DataLoader with pin_memory + workers
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4 if device.type == "cpu" else 0,  # MPS doesn't support multiprocessing
    pin_memory=device.type == "mps",
)
```

**Note:** `pin_memory=True` only works on CPU; for MPS the tensor is already in unified memory, so it has no effect. But `num_workers=0` on MPS is required because multiprocessing on MPS is not supported.

### 2.2 Scaling Data: 500 Maps

With the optimizations above, 500 maps is trivial. Here's the math:

| Metric | 100 maps | 500 maps | Notes |
|--------|----------|----------|-------|
| Total frames | ~1.5M | ~7.5M | 5× |
| Batch size 16 | ~94 batches/epoch | ~469 batches/epoch | 5× |
| Time per epoch (optimized) | ~10s | ~50s | 5× |
| 100 epochs total | ~17 min | ~1.4 hours | Acceptable |

**No architecture changes needed** for 500 maps. The model is small enough.

### 2.3 Scaling Further: 2,000+ Maps (Full Master/Expert)

If you want to train on the full Master+Expert set (~3.8K maps):

| Concern | Status | Action |
|---------|--------|--------|
| Memory (dataset) | ~60GB raw features | Use **lazy loading** with an LRU cache, or **ChunkedDataset** |
| Memory (model) | ~14MB | No problem |
| Compute time | ~4–5 hours/100 epochs with MPS | Acceptable for overnight runs |
| Disk I/O | Single HDD read will bottleneck | Ensure dataset is on **SSD** (currently on external volume) |

**Recommendation for full corpus:**
1. Move `dataset/features/` and `dataset/audio_features/` to the Mac Mini's internal SSD if possible
2. Implement a **dataset shard** or **generator-based DataLoader** that streams from disk
3. Consider **mixed precision (float16)** training on MPS if supported — halves memory and often speeds up

---

## 3. Architecture Scaling Considerations

### Current Model Size

```
Conv1D Encoder:  80 -> 128 × 4 layers  ≈ 200K params
LSTM Decoder:    128 -> 256 × 2 layers  ≈ 800K params
Heads:           256 -> 2 + 4 + 1      ≈ 200K params
Total:                               ≈ 1.2M params
```

### 3.1 Does the model need to grow for 500 maps?

**No.** 1.2M parameters is fine for 500 maps. Overfitting risk at 100 maps is the current concern; at 500 with real audio, underfitting is more likely.

### 3.2 When to scale the model

| Dataset Size | Model Params | Notes |
|-------------|--------------|-------|
| 100 maps | 1.2M | Current, slight overfitting expected |
| 500 maps | 1.2–3M | Add 1–2 LSTM layers or increase hidden dim |
| 2K maps | 3–5M | Add Transformer encoder or deeper Conv1D |
| 5K maps | 5–10M | This is where Transformers shine |

### 3.3 Suggested architecture evolution

For 500 maps, one change is worth trying: **Transformer decoder instead of LSTM**.

```python
class TransformerDecoder(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=256, num_layers=4, nhead=8):
        super().__init__()
        self.pos_enc = PositionalEncoding(latent_dim, max_len=15000)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.presence_head = nn.Linear(latent_dim, 2)
        self.position_head = nn.Linear(latent_dim, 4)
        self.rail_head = nn.Linear(latent_dim, 1)

    def forward(self, x, mask=None):
        x = self.pos_enc(x)
        out = self.transformer(x, src_key_padding_mask=mask)
        return {
            "presence_logits": self.presence_head(out),
            "position_pred": self.position_head(out),
            "rail_logits": self.rail_head(out),
        }
```

**Pros:**
- Better long-range dependencies (notes spaced seconds apart)
- Parallelizable (faster on MPS than sequential LSTM)
- Scales gracefully to longer songs

**Cons:**
- Memory scales quadratically with sequence length: O(T²)
- At 15K frames, this is ~225M attention entries → ~900MB per sample → **not feasible on M4**

**Solution:** Use **local attention** or **sliding window** (e.g., window size 512, stride 256):

```python
class SlidingWindowTransformer(nn.Module):
    def __init__(self, ...):
        self.window_size = 512
        self.stride = 256

    def forward(self, x):
        # Process in overlapping chunks
        T = x.shape[1]
        outputs = []
        for start in range(0, T, self.stride):
            end = min(start + self.window_size, T)
            chunk = x[:, start:end]
            out = self.transformer(chunk)
            outputs.append(out)
        # TODO: stitch overlapping regions
        return torch.cat(outputs, dim=1)
```

**Verdict:** Keep LSTM for now. Transformers are promising but need careful sequence-length management. Revisit when moving to 2K+ maps.

---

## 4. Data Pipeline Scaling

### 4.1 Feature Precomputation

Feature extraction is already done (`dataset/features/*.npz`), but audio feature extraction is on-demand per training run. For 500 maps, precompute audio features for ALL maps:

```bash
# Run once, save for all future training
python scripts/extract_audio_batch.py \
    --parsed-dir dataset/parsed \
    --extracted-dir dataset/extracted \
    --output-dir dataset/audio_features
```

This ensures every training run skips the expensive librosa loading/resampling.

### 4.2 Difficulty Augmentation

Currently training on Hard. For scaling, train on **all difficulties simultaneously** with a difficulty embedding:

```python
class DifficultyAwareModel(BaselineBeatmapModel):
    def __init__(self, ...):
        super().__init__()
        self.difficulty_emb = nn.Embedding(4, 64)  # Normal, Hard, Expert, Master

    def forward(self, audio_features, difficulty_idx):
        latent = self.encoder(audio_features)  # (B, T, D)
        diff = self.difficulty_emb(difficulty_idx)  # (B, 64)
        diff = diff.unsqueeze(1).expand(-1, T, -1)  # (B, T, 64)
        combined = torch.cat([latent, diff], dim=-1)  # (B, T, D+64)
        return self.decoder(combined)
```

This turns 4.6K maps into **4.6K × 4 = 18K effective training examples**, massively increasing data diversity.

**BUT:** Memory per sample increases by ~64 bytes/frame. At 15K frames, that's **~1 MB extra per sample** — negligible.

### 4.3 Multi-Difficulty Training

Recommended approach:
1. Start with Hard only (as now) — validate pipeline
2. Add Expert (37.8% of corpus) — increase to ~2.5K maps
3. Add Master (44.7%) — full dataset
4. Add Normal — not critical, but nice for completeness

---

## 5. Training Schedule Recommendation

### Phase A: Validate Overfit (Current)
- **100 maps**, Hard, synthetic audio
- **Goal:** 100% training recall, <50ms timing error
- **Duration:** ~3 hours (current CPU run)
- **Pass criteria:** Model should memorize training set

### Phase B: Generalization Test (Next)
- **100 maps**, Hard, **real audio features**
- **Train/val split:** 80/20 (not same data for both)
- **Goal:** >80% val recall, <100ms timing error
- **Duration:** ~3 hours
- **Pass criteria:** Model generalizes, not just memorizes

### Phase C: Scale to 500 Maps
- **500 maps**, Hard + Expert
- **Batch size:** 16
- **Epochs:** 50–100
- **Goal:** >70% val recall, <150ms timing error
- **Duration:** ~1.5 hours with MPS optimizations
- **Pass criteria:** Consistent generation quality

### Phase D: Full Corpus (2K+ Maps)
- **2,000 maps**, all difficulties
- **Batch size:** 32
- **Architecture:** Try Transformer decoder with sliding window
- **Duration:** Overnight run (~8–12 hours)
- **Goal:** >65% val recall, <200ms timing error (this is production-ready)

---

## 6. Critical Blockers to Address

### Blocker 1: MPS Not Engaged
**Likelihood:** High
**Check:** Run `python3 -c "import torch; print(torch.backends.mps.is_available())"`
**Fix:** If False, ensure PyTorch ≥ 2.0 is installed with MPS support. If True but still slow, use `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` to eliminate memory fragmentation issues.

### Blocker 2: Audio Feature Extraction Speed
librosa.load() + melspectrogram() is single-threaded and slow.
- Consider using **torchaudio** (faster on MPS, but less flexible)
- Or **precompute all audio features once** and never run extraction during training

### Blocker 3: Variable-Length Padding Overhead
Currently padding every sample to 15,000 frames (5 min max). Most songs are 2–3 min (~6K–9K frames). **50% of each batch is padded zeros**.

**Fix:** Sort by length within batches, use **bucketing** (group similar lengths together), and compute with variable lengths. Or use **PackedSequence** with LSTM (but this breaks with Conv1D encoder).

**Simpler fix:** Add a `max_length` parameter to dataset that clips long sequences. Most songs don't need 15K frames. Set it to 10,000 (3.3 min) and see if any songs are cut.

---

## 7. Summary Action Plan

| Priority | Task | Expected Impact | Time |
|----------|------|-----------------|------|
| 🔥 **P0** | Verify MPS is active + increase batch size | 10× speedup | 5 min |
| 🔥 **P0** | Pre-load dataset to RAM | 1.5× throughput | 20 min |
| **P1** | Run holdout evaluation on current best_model.pt | Know if it generalizes | 10 min |
| **P1** | Precompute all audio features | 2× per-epoch speed | ~30 min (one-time) |
| **P2** | Scale to 500 maps (Hard+Expert) | Validate general scaling | ~2 hours |
| **P2** | Try Transformer decoder with sliding window | Better long-range | ~3 hours |
| **P3** | Full corpus training (2K+ maps) | Production-ready model | Overnight |

---

## 8. Quick Reference: One-Command Runs

```bash
# Check MPS
python3 -c "import torch; print(torch.backends.mps.is_available())"

# Run holdout evaluation (after saving the new script)
cd /Volumes/Second-Brain-1/AI/Synth
python3 src/evaluation/evaluate_holdout.py \
    --checkpoint models/checkpoints/best_model.pt \
    --features-dir dataset/features \
    --audio-features-dir dataset/audio_features \
    --parsed-dir dataset/parsed \
    --output-dir evaluation/holdout \
    --num-maps 50 \
    --train-split 0.8

# Scale to 500 maps with optimized settings
python3 src/training/train_baseline.py \
    --features-dir dataset/features \
    --audio-features-dir dataset/audio_features \
    --difficulty Hard \
    --num-maps 500 \
    --epochs 100 \
    --batch-size 16 \
    --device mps

# Precompute audio for ALL maps (one-time)
python3 scripts/extract_audio_batch.py \
    --extracted-dir dataset/extracted \
    --output-dir dataset/audio_features \
    --parsed-dir dataset/parsed
```
