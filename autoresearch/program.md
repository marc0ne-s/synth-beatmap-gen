# autoresearch-synth

Autonomous research for SynthRiders AI beatmap generation.

## Setup

To set up a new experiment, confirm:

1. **Agree on a run tag**: Use a date-based tag (e.g. `apr27`). The branch `autoresearch/synth-<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/synth-<tag>` from current master.
3. **Read in-scope files** for full context:
   - `autoresearch/synth_train.py` — the file you modify. Model assembly, training loop, optimizer.
   - `src/models/transformer.py` — TransformerCausalDecoder architecture.
   - `src/models/baseline.py` — Conv1D+LSTM baseline (frozen, read-only).
   - `src/data/streaming_loader.py` — Data pipeline (read-only).
4. **Check data exists**: Verify `/Volumes/Second-Brain-1/AI/Synth/dataset/features/` has `.npz` files and `/dataset/audio_features/` has `.npz` files. If not, training won't work.
5. **Initialize results.tsv**: Create `autoresearch/results.tsv` with header row.
6. **Confirm and go**: Confirm setup looks good.

## Experimentation

Each experiment trains for a **fixed budget of 5 epochs** (~30-45 minutes wall clock on MPS). Launch as: `python autoresearch/synth_train.py`.

**What you CAN do:**
- Modify `autoresearch/synth_train.py` — model architecture, optimizer, hyperparameters, training loop, fusion strategy.
- Add new modules (e.g. different fusion strategies, attention mechanisms)
- Modify the `SynthConfig` dataclass (learning rates, batch sizes, model dims)

**What you CANNOT do:**
- Modify files outside `autoresearch/synth_train.py` (unless you're adding new modules in `autoresearch/`)
- Install new packages
- Modify the evaluation harness (`compute_metrics` and `compute_masked_loss` in synth_train.py)
- Change the data pipeline

**The goal: maximize the score.**

Score = `val_recall_50 * (1 / (1 + val_position))`

This balances:
- High recall (catching more ground truth notes within 50ms)
- Low positional MSE (accurate spatial placement)

**VRAM** is a soft constraint. MPS can OOM. If training crashes, reduce batch_size or model dims.

**Simplicity criterion**: All else equal, simpler is better. A small score improvement from deleting code is better than from adding complexity.

## Output format

Once training finishes, it prints:

```
==================================================
RESULTS
==================================================
val_recall_50:    0.536100
val_position:     19.822000
val_precision_50: 0.378800
val_loss:         1.234567
score:            0.025000
peak_vram_mb:     45060.2
==================================================
```

Extract the score:
```bash
grep "^score:" run.log
```

## Logging results

When an experiment is done, log it to `autoresearch/results.tsv` (tab-separated).

```
commit	score	recall_50	position_mse	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. score (0-1 range, higher is better)
3. recall_50 (e.g. 0.536)
4. position_mse (e.g. 19.822)
5. peak memory in GB, round to .1f
6. status: `keep`, `discard`, or `crash`
7. short text description

Example:
```
commit	score	recall_50	position_mse	memory_gb	status	description
a1b2c3d	0.0254	0.536	19.822	4.5	keep	baseline hybrid cascade
d2e3f4g	0.0278	0.552	18.500	4.6	keep	increase coarse_proj LR to 8e-4
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/synth-apr27`).

LOOP FOREVER:

1. Look at the current git state
2. Tune `synth_train.py` with an experimental idea
3. git commit
4. Run the experiment: `python autoresearch/synth_train.py > run.log 2>&1`
5. Read results: `grep "^score:" run.log`
6. If grep is empty, run crashed. Check `tail -50 run.log` for traceback
7. Record results in `results.tsv`
8. If score improved (higher), keep the commit
9. If score equal or worse, git reset

The idea is you are an autonomous researcher trying things out. If they work, keep. If they don't, discard.

**Timeout**: Each experiment should take ~30-45 minutes. If a run exceeds 90 minutes, kill it and treat as failure.

**Crashes**: If OOM, try reducing batch_size or d_model. If code errors, fix if easy; if the idea is fundamentally broken, log "crash" and move on.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. Run autonomously. If you run out of ideas, think harder — re-read the code, re-read the transformer.py, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you.

## Key context

### Phase 10 Diagnostic (baseline for comparison)
- Recall@50ms: 53.61%
- Position MSE: 19.822
- Precision@50ms: 37.88%
- **Score: ~0.0254**

### What Phase 10 learned
The transformer was trained with **teacher forcing** — always fed ground truth history. At inference, when this was removed, the model's own noisy predictions caused compounding spatial drift (MSE shot up from training to inference).

### The Fix (Phase 11)
Hybrid Cascade architecture:
1. Freeze the old Conv1D+LSTM baseline (provides robust coarse predictions)
2. Transformer learns to **refine** LSTM outputs (residual correction)
3. Inference: LSTM generates → Transformer refines → much more stable

### Your Job
Find the best way to fuse LSTM coarse memory into the Transformer. Ideas to try:

**Fusion strategies:**
a. Simple addition (current): `x = x + coarse_proj(coarse_memory)`
b. Concatenation then projection
```python
x = torch.cat([x, coarse_proj(coarse_memory)], dim=-1)
x = fusion_proj(x)  # project back to d_model
```
c. Cross-attention: Query=x, Key/Value=coarse_memory
d. Gated fusion: `x = gate * x + (1-gate) * coarse_memory`

**Layer placement:**
a. Fuse before the decoder blocks (current)
b. Fuse inside each NativeCausalBlock
c. Fuse only at the output heads

**Learning rate strategies:**
a. High LR for coarse_proj only (current: 5e-4 vs 1e-4)
b. Higher LR for all new params
c. Use LR finder or warm restarts
d. Try Adam vs AdamW vs Muon

**Architecture knobs:**
a. Number of layers (currently 4)
b. d_model size (currently 256)
c. Kernel size for CausalConv1d (currently 7)
d. Adding/removing CausalConv1d bridge
e. Try different positional encodings

**Loss / Training:**
a. Current: Focal γ=2.0, pos_weight=1.5
b. Try different gamma values
c. Try without focal (just weighted BCE)
d. Adjust position loss weight (currently 10.0)
e. Adjust velocity loss weight (currently 25.0)
f. Try curriculum learning (easy → hard difficulties)

## Target Scores

| Metric | Phase 10 Baseline | Target | Stretch |
|--------|-------------------|--------|---------|
| Recall@50ms | 53.61% | >65% | >70% |
| Position MSE | 19.822 | <5.0 | <1.0 |
| Score | ~0.025 | >0.08 | >0.12 |

**Phase 11 success = Score > 0.08** (Recall ~65%, MSE ~5.0)
**Stretch goal = Score > 0.12** (Recall ~70%, MSE ~1.0)

## Notes

- Training is on MPS (Apple Silicon). If MPS crashes, fall back to CPU or reduce batch size.
- The dataset is 2,500 maps (Hard/Expert/Master). Training subset is 80/20 split.
- Batch size is 8 with gradient accumulation 4 (effective 32). This is stable on MPS.
- Each epoch processes ~2,000 maps. At ~30s/batch, this is ~750 batches = ~6 hours/epoch... wait, that's too slow.

Actually: The `num_maps=2500` is the dataset. The batch processes chunks of sequences, not full songs. Each batch is fast (~1-2s on MPS). At 200 maps per epoch, each epoch is ~20 minutes.

Wait, I should clarify: The streaming loader processes batches of (B, T, F) where T is time steps (up to 4096 due to attention window). Multiple chunks per song. An epoch is 2500 maps worth of chunks.

**Bottom line**: 5 epochs is ~30-60 minutes.

Good luck, researcher.
