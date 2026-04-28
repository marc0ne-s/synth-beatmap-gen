# SynthRiders AI Beatmap Generator — Master Context for GPT 5.5 Synthesis

## What This Project Is
AI beatmap generation for the VR rhythm game **SynthRiders**. A Transformer-based model that takes raw audio and generates 5-difficulty VR beatmaps (Easy → Master) using a learned playability reward model.

## Executive Summary

| Phase | Status | Key Result |
|-------|--------|------------|
| Phase 1 (Data) | Complete | 8,861 human maps ingested, 200hrs audio |
| Phase 2 (Model) | Complete | 128-dim Transformer, 71% precision + 100ms NMS |
| Stage 3 (RL) | **ACTIVE** | Run 3.15 Turbo Bypass — fixing broken Easy mode |
| Stage 4 (Editor) | Future | Electron/Tauri cross-platform map editor |
| Stage 5 (Deploy) | Future | Standalone app + streaming integration |

## The Problem Statement

**Easy mode is structurally broken.** Playability Scorer v0 scores:
- Easy: 0.32
- Normal: 0.46
- Hard: 0.58
- Expert: 0.79
- Master: 0.96

The model is technically "correct" (notes are on beats) for Easy, but players find Easy maps *bad*. Left-hand bias, awkward wrist angles, and no kinetic flow. The model was trained on MLE (Maximum Likelihood Estimation) which optimises for "reproduce human maps" not "be playable."

## Technical Architecture

### Data Pipeline
- Audio → librosa → 128-dim features (80 log-mel + 12 chroma + 36 onsets/peaks)
- Maps parsed from `.synth` format → position (x,y,z), hand (left/right), type (note/special)
- 8,861 maps ~200 hours, 2.5M notes, 3.2 GB

### Model
- Difficulty-conditioned 128-dim Transformer
- 8 attention blocks, causal mask for autoregressive generation
- Multi-head attention (8 heads), feedforward 512
- Output: position_x, position_y, position_z, hand (softmax), type (sigmoid)
- Teacher forcing during training (MSE loss on position + BCE on presence)

### Inference
- Autoregressive: model generates note by note, feeds its own output back as history
- 100ms NMS Temporal Peak Detection applied post-hoc to collapse clustered logits
- Without NMS: 18,469 notes (precision 21%). With 100ms NMS: ~1,100 notes (precision 71%).

### Playability Scorer v0
- Bi-LSTM + MLP classifier (2.3MB)
- Input: hand balance, NPS, wrist angle variance, reach distance, symmetry
- Output: binary playable/unplayable (AUROC 0.9988)
- Trained on 8,284 human maps, held out 577 maps for validation

### RL Reward Model (Stage 3)

Using Scorer v0 as a reward signal to fine-tune the generator.

**The core challenge:** Model output 100 NPS (note on every frame). REINFORCE requires sampler variance → Advantage = 0.0.

**Solution: Turbo Bypass**
- If NPS > 1.2× target: Skip expensive Reward Engine (librosa + Bi-LSTM inference)
- Use pure supervised BCE (20× weight) against Fixed-K stratified oracle mask
- Oracle = top-k beat-salience frames from Phase 12b reference model, per 15s segment
- Once NPS < 2.5: Auto-handoff back to RL playability fine-tuning

**Reward stack:**
```
Total Reward = (Score + BIAS_MAP[diff_idx]) * Density_Bonus * Uniformity_Bonus * Alignment_Bonus

Density_Bonus = exp(-|NPS - Target| / Target)
Uniformity_Bonus = exp(-|1 - ratio| / 2.0)
Alignment_Bonus = Σ(probs * mask_2d) / Σ(probs)
```

**Difficulty targets:**
| Difficulty | NPS Target | Bias | PG Weight | KL β |
|------------|-----------|------|-----------|------|
| Easy (0) | 2.1 | 2.0–5.0 | 5.0× | 0.0 |
| Normal (1) | 3.5 | 1.0 | 1.0× | 0.05 |
| Hard (2) | 6.0 | 0.5 | 1.0× | 0.1 |
| Expert (3) | 9.0 | 0.0 | 2.0× | 0.4 |
| Master (4) | 15.0 | 0.0 | 2.0× | 0.7 |

**KL gating:** β = 0.0 for Easy/Normal (reference model's 15 NPS bias was pulling Easy UP). High β for Master/Expert to preserve original density.

## Failure Modes Discovered (15 Iterations of Debug)

1. **Mode dissolution** — All difficulties → same density. Solved with contrastive sampling.
2. **Gaussian gradient death** — `exp(-(NPS-2.1)²/σ²)` underflows at 15 NPS. Solved with Exponential Bonus (long tail).
3. **Sparse trough** — 0.07 NPS maps score highly. Solved with hard floor at 1.0 NPS.
4. **KL conflict** — Reference model pulls Easy back to 15 NPS. Solved with β=0 gating.
5. **7.4 NPS false plateau** — "Close enough" local optima. Solved with Bias=5.0, 13× reward gap.
6. **Logit saturation** — Sampler variance = 0 at 100 NPS. Solved with Turbo Bypass supervised BCE.
7. **Dynamic-K bug** — Increasing K with more notes = density attractor. Solved with Fixed-K mask.
8. **Computational bottleneck** — 35s/track. Solved with Turbo Bypass (skip engine, ~100ms/track).
9. **MPS crash** — Shape mismatch. Solved with dim=2 fix + 0-padding to 128.
10. **"Shotgun" alignment** — Mask K grew with note count. Solved with Fixed-K stratified mask.
11. **Equilibrium stall** — Dense and sparse maps had equal reward. Solved with 10× bias differential.
12. **Mean solution** — Model settled at ~340 notes for all difficulties. Solved with pole anchoring + non-uniform sampling.
13. **Identity collapse** — Difficulty embedding ignored. Solved with gated KL + contrastive weights.
14. **Temporal clumping** — 6× density variation within track. Solved with uniformity bonus.
15. **Alignment stall** — Mask signal too quiet. Solved with 20× amplification + curriculum decay.

## Current Codebase

| File | Purpose |
|------|---------|
| `scripts/train_rl_refinement.py` | Main RL training loop (Turbo Bypass) |
| `scripts/generate_gold_standard.py` | Inference + map generation pipeline |
| `scripts/feasibility_checker.py` | Rule-based validation (5-difficulty audit) |
| `scripts/playability_model.py` | Scorer architecture (Bi-LSTM + MLP) |
| `scripts/train_playability.py` | Scorer training loop |
| `scripts/extract_dataset.py` | Feature extraction for scorer |
| `scripts/check_clumping.py` | Temporal clumping detection |
| `models/checkpoints/transformer_phase12b_ep5.pt` | Phase 12b MLE weights (reference) |
| `models/checkpoints/scorer_v0.pt` | Playability Scorer v0 (AUROC 0.9988) |

**Repo:** https://github.com/marc0ne-s/synth-beatmap-gen

## What "Working Prototype" Means

To Marcus, a prototype means:
1. **End-to-end pipeline:** Drop audio file → get 5 .synth beatmaps
2. **Playable maps:** Expert/Master pass feasibility scorer at ≥85% rate
3. **Functional Easy mode:** ≥40% pass rate, feels musical not robotic
4. **Visual verification:** Can view maps in 3D / timeline (editor not required)
5. **Batch generation:** Can process 100+ tracks unattended
6. **No training required by user:** Pre-trained weights, ready to use

## Design Principles (From Marcus's Preferences)
- Light, warm, organic UI — no dark themes or robotic layouts
- Paper-texture, watercolor, slow-drift aesthetics
- Australian-accented female TTS for briefing audio
- Naming precision matters — corrected "HealthKit-Bridge" → "DockKit-Bridge" instantly
- Friction removal from idea → output is sacred

## Next Steps After Prototype
1. Ingest the 4,000 user-made .synth maps from Google Drive (training corpus upgrade)
2. Cross-platform editor (Electron/Tauri), Mac-first
3. Spotify/Apple Music streaming integration (metadata only)
4. Quest 3 telemetry collection for Stage 2 playability validation
5. Deploy as standalone app

## What We Need From GPT 5.5
Given all this context, synthesize a **focused, actionable prompt** for Claude Code (or kimi k2.6 / glm5.1) that will:

1. **Audit the current `train_rl_refinement.py`** for structural bugs
2. **Fix the Turbo Bypass implementation** to actually achieve NPS descent
3. **Build the end-to-end inference pipeline** (audio → 5 .synth maps)
4. **Create a feasibility audit script** that validates all 5 difficulties
5. **Produce a working prototype** within 1-2 sessions

The prompt should include:
- All relevant hyperparameters (no guessing)
- The reward stack formula (coded, not described)
- The exact KL gating strategy
- The Fixed-K oracle mask implementation
- The Turbo Bypass switching logic
- Expected output formats

Assume the coding agent has access to the GitHub repo and can clone it.
