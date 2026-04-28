# SynthRiders Stage 3: RL Refinement — Antigravity Handoff
## Generated: 2026-04-28 20:45
## Status: Run 3.15 Turbo Bypass ACTIVE
## Next Morning: Verify density snap convergence

---

## Executive Summary

We are deep into Stage 3: RL Generator Integration for SynthRiders AI beatmap generation. The goal is to use the Playability Scorer v0 (AUROC 0.9988) as a reward model to refine the Phase 12b Transformer Generator, specifically fixing the structurally broken Easy mode (0.32 scorer vs 0.96 Master).

After extensive iteration through Run 3.0 → Run 3.15, the project has arrived at a working architecture called **"Turbo Bypass"** that achieves ~100ms per track training speed by bypassing the expensive Reward Engine (librosa + Bi-LSTM Scorer) during the initial density-descent phase.

## Current State: Run 3.15 Turbo Bypass

### Active Architecture

| Component | Setting | Notes |
|-----------|---------|-------|
| **Window** | 20s crop (2,000 frames) | Ultra-fast iteration |
| **Supervised Weight** | 20.0× | BCE against Fixed-K stratified oracle mask |
| **Bypass Mode** | NPS > 1.2× target | Skips Reward Engine + Scorer entirely |
| **Handoff Trigger** | NPS < 2.5 for Easy | Auto-re-enables RL fine-tuning |
| **Per-track time** | ~100ms | Down from 35s (350× speedup) |
| **Current NPS** | 100.0 → saturating every frame | Expected descent: 100→50→10→2.1 over next 2-3 min |

### The Key Insight: Logit Saturation

The model had fully saturated to outputting a note on every single frame (100 NPS, i.e. 1 note per 10ms hop). This is why earlier REINFORCE attempts failed — the sampler had zero variance, so Advantage = 0. The Turbo Bypass bypasses REINFORCE entirely during the density-crash phase and uses pure supervised BCE against a sparse oracle mask (only ~0.5% of frames are "1"s) to directly pull logits down.

### Oracle Mask Design (FINAL)

The Fixed-K Stratified Mask:
```python
K_global = int(DENSITY_TARGETS[diff_idx] * duration_s)  # e.g. 30 for Easy
K_per_seg = max(1, K_global // num_segments)
# Per-segment torch.topk on ref model's salience
# Selects highest-confidence beat frames per ~15s segment
```

This prevents chorus-clumping and gives musical structure even at extreme sparsity.

### Difficulty-Aware Architecture (FINAL)

| Difficulty | Density Target | Bias | PG Weight | KL β |
|------------|---------------|------|-----------|------|
| Easy (0) | 2.1 NPS | 2.0-5.0 | 5.0× | 0.0 |
| Normal (1) | 3.5 NPS | 1.0 | 1.0× | 0.05 |
| Hard (2) | 6.0 NPS | 0.5 | 1.0× | 0.1 |
| Expert (3) | 9.0 NPS | 0.0 | 2.0× | 0.4 |
| Master (4) | 15.0 NPS | 0.0 | 2.0× | 0.7 |

### Reward Stack (FINAL)

```
Total Reward = (Score + BIAS) * Density_Bonus * Uniformity_Bonus * Alignment_Bonus

Density_Bonus = exp(-|NPS - Target| / Target)  # Prevents flatlining
Uniformity_Bonus = exp(-|1 - ratio| / 2.0)     # Prevents temporal clumping
Alignment_Bonus = Σ(probs * mask_2d) / Σ(probs) # Precision vs oracle
```

### Files & Checkpoints

| File | Purpose |
|------|---------|
| `scripts/train_rl_refinement.py` | Main Run 3.15 training loop (Turbo Bypass) |
| `models/checkpoints/phase12b_ep5.pt` | Original MLE Phase 12b weights (reference) |
| `models/checkpoints/scorer_v0.pt` | FeasibilityScorer v0 (AUROC 0.9988) |
| `scripts/playability_model.py` | Scorer architecture (Bi-LSTM + MLP) |
| `scripts/extract_dataset.py` | Feature extraction for scorer |
| `scripts/feasibility_checker.py` | Rule-based checker for audits |
| **Next checkpoint** | `ep1.pt` from Run 3.15 (in ~2-3 min) |

### Current Known Risks

1. **Trough risk**: Below 1.0 NPS the density bonus cratered in earlier runs. The hard floor should prevent this, but monitor.
2. **Mid-tier compression**: Normal/Hard might cluster near Easy if Easy's bias dominates shared parameters.
3. **Over-smoothing**: Supervised BCE might teach the model to spread notes evenly across mask frames rather than clustering on strongest beats. Check alignment after handoff.
4. **Shared encoder drift**: During Turbo Bypass, only presence head gets gradients. Full-track inference on Master should be run every 50 tracks to verify shared encoder hasn't decayed.

### What To Do Next Morning

**Check 1: Density Snap Confirmation**
- Open latest log: `grep "DEBUG.*NPS=" training.log | tail -20`
- Verify Easy NPS has dropped below 5.0 (target: 2.1)
- If still ≥ 8.0 after 50+ tracks, the supervised signal isn't penetrating → discuss logit reset

**Check 2: Master Stability**
- Run full-track inference on 5 Master maps from the latest checkpoint
- Verify Master NPS ≥ 12 and Score ≥ 0.9
- If collapsed, shared encoder drift occurred during bypass

**Check 3: Alignment Quality**
- Check `Align=` values in logs for tracks with NPS 2-5
- Should see ≥ 0.3 (model placing notes on beat salience)
- If < 0.1, oracle mask or audio encoder issue

**Check 4: Handoff to RL**
- When Easy NPS < 2.5, verify the handoff logic re-enables Reward Engine
- First RL tracks should show Score > 0 and non-zero Advantage
- If Score stays at 0.0, alignment needs more epochs before RL takeover

**If Turbo succeeds → Phase 12b.Final Audit**
Run the 5-difficulty feasibility audit:
```bash
python scripts/feasibility_checker.py --input evaluation/phase12b_rl/ --output audit.json
```

Target gates:
- Easy: ≥ 40% pass, NPS 1.8–3.5
- Normal: ≥ 60% pass, monotonic step up from Easy
- Hard: ≥ 75% pass
- Expert: ≥ 85% pass
- Master: ≥ 90% pass, NPS ≥ 12

---

## Phase History Summary

- **Phase 10**: Collapsed at inference (53.6% recall). Root cause: teacher forcing.
- **Phase 11**: Hybrid cascade. KILLED. LSTM choked the signal.
- **Phase 12a**: Head reset. KILLED. Destroyed presence calibration.
- **Phase 12b**: Fresh training on 8,861 maps. Deployed 100ms NMS post-hoc.
- **Stage 3 Scorer v0**: Bi-LSTM + MLP, AUROC 0.9988.
- **Run 3.x series**: 15 iterations of RL reward engineering to solve:
  - Mode dissolution (all difficulties → same density)
  - Logit saturation (100 NPS, no sampler variance)
  - Gaussian gradient death (exp(-83) underflow)
  - Sparse trough (empty maps rewarded)
  - KL conflict (reference model pulling Easy back up to 15 NPS)
  - 7.4 NPS false plateau ("close enough" local optima)
  - Computational bottleneck (35s per track)

---

## Context for Next Agent

**Marcus's working style:**
- Drives hard on weekends and nights, burns out after ~8-10 hours of deep work
- Expects EOD handoffs and morning briefings
- Likes technical summaries but prefers "what's the next action" framing
- Frustrated by slow iteration loops — Turbo Bypass was built specifically to fix this
- Will sometimes pivot hard (e.g. Run 3.14 "Hurricane") when gentle approaches stall
- Musical ear is sharp — if maps "don't feel right" he'll catch it

**Project repo:** `~/synth-beatmap-gen` or local path
**Key scripts:** `scripts/train_rl_refinement.py`, `scripts/generate_gold_standard.py`, `scripts/feasibility_checker.py`
**Scorer model:** `models/checkpoints/scorer_v0.pt`
**Reference model:** `models/checkpoints/transformer_phase12b_ep5.pt`

If Run 3.15 is still running in the morning, verify the NPS descent curve and proceed with the handoff audit. If it's completed, run the Phase 12b.Final Audit and report results.

If Marcus asks about Stage 2 (Quest 3 Telemetry collection), remind him Stage 3 is the current priority and telemetry is a future enhancement.

---
*Handoff generated by HerBB — SynthRiders Chief of Staff*
