# SYNTHRIDERS AI — RELAY PROMPT: PHASE 10

> **Context:** This document serves as the baton for the HerBB ↔ Antigravity relay.  
> **Current State:** Phase 9 Structural Redesign COMPLETE — Transformer training actively running.  
> **Mission:** Continue to Phase 10 validation & evaluation.

---

## 🎯 PRIMARY OBJECTIVE

Validate trained TransformerCausalDecoder against ground truth and prepare for inference deployment.

---

## 📊 CURRENT SYSTEM STATE

### Training Status (AS OF 2026-04-27 ~14:00)
- **Active Training Process:** PID 9456 running since 13:57 (8+ hours)
- **Latest Checkpoint:** `transformer_pilot_ep5.pt` (12:49 PM)
- **Training Script:** `src/training/train_transformer_pilot.py`
- **Architecture:** TransformerCausalDecoder (256d, 4 layers, hybrid Conv1D attention)
- **Corpus:** 2,500 maps (Hard, Expert, Master difficulties)
- **Loss Function:** Focal Loss (γ=2.0, pos_weight=1.5) + Huber position + scaled velocity
- **Device:** MPS (Apple Silicon) — running stable

### Checkpoints Available
```
models/checkpoints/
├── transformer_pilot_ep1.pt    (14:08)
├── transformer_pilot_ep2.pt    (14:14)
├── transformer_pilot_ep3.pt   (14:21)
├── transformer_pilot_ep4.pt    (12:41 - phase transition)
├── transformer_pilot_ep5.pt    (12:49 - latest)
└── [ep6.pt potentially in progress from PID 9456]
```

### Architecture Lock-in
- `CausalConv1d` with `kernel_size=7` (strict left-padding, no future leak)
- `NativeCausalBlock` = Causal Self-Attn → Causal Cross-Attn → Conv Bridge → FFN
- Output heads: presence (2), position (4), velocity (2) segregated
- d_model=256, num_layers=4, d_audio=128, d_target=8

### Loss Components Proven Working
```python
total_loss = focal_presence + (position * 10.0) + (velocity * 25.0)
```

---

## ✅ PHASE 10 CHECKLIST

### Task 1: Training Status Confirm
- [ ] Check if PID 9456 has completed → identify latest checkpoint
- [ ] If crashed: capture error, resume from ep5.pt
- [ ] If converged: identify best epoch by validation loss

### Task 2: Quantitative Evaluation Run
- [ ] Execute `scripts/evaluate_checkpoint.py` (create if missing) vs ep5.pt
- [ ] Metrics to capture:
  - **Recall@50ms** — what % of ground truth notes are predicted within 50ms
  - **Precision@50ms** — what % of predictions match ground truth within 50ms
  - **Position MSE** — for correctly-timed predictions, position error
  - **Hand accuracy** — left/right prediction correctness
  - **Difficulty breakdown** — per-difficulty metrics (Easy/Normal/Hard/Expert/Master)
- [ ] Export results to `evaluation/phase10_quantitative.json`

### Task 3: Latent Space Re-Visualization
- [ ] Re-run `scripts/visualize_latent.py` on FINAL checkpoint (not ep5)
- [ ] Verify PCA dispersion is maintained (not re-collapsing to singularity)
- [ ] Compare ep1 vs ep5 vs final latent distributions

### Task 4: Qualitative Beatmap Generation
- [ ] Select 3 test tracks (variety: low BPM ballad, mid BPM EDM, high BPM DnB)
- [ ] Run inference: `python src/inference/generate_beatmap.py --checkpoint [BEST].pt`
- [ ] Generate `.synth` files for all 5 difficulties per track
- [ ] Document qualitative assessment in `evaluation/phase10_qualitative.md`

### Task 5: Comparative Analysis
- [ ] Compare Transformer vs Baseline (Conv1D+LSTM from `best_model.pt`)
- [ ] Note count accuracy, timing precision, position variance
- [ ] Document in `evaluation/baseline_vs_transformer.md`

---

## 🔧 CRITICAL TECHNICAL CONSTRAINTS

### MPS (Apple Silicon) — WORKING
- **Status:** Stable with `torch.amp.autocast(device_type="mps", dtype=torch.bfloat16)`
- **Known Issue:** Old "division error" (torch.div) was fixed in newer PyTorch
- **Fallback:** CPU mode if MPS crashes — but should not be needed

### Data Paths (DO NOT CHANGE)
```
Features: /Volumes/Second-Brain-1/AI/Synth/dataset/features
Audio: /Volumes/Second-Brain-1/AI/Synth/dataset/audio_features
Raw .synth: /Volumes/Second-Brain-1/Meta Quest/Synth/ (8,301 files)
Parsed: /Volumes/Second-Brain-1/AI/Synth/dataset/parsed/ (4,638 maps)
```

### File Format (Locked)
- **Feature files:** PyTorch tensors via `torch.save()` — shape `(T, 8)`
- **Audio features:** Mel spectrograms ` (T, 128)` — real extracted, not synthetic
- **Output .synth:** ZIP containing `beatmap.meta.bin` (JSON with BOM)

---

## 📁 WHERE TO CREATE ARTIFACTS

All Phase 10 outputs go here:
```
/Volumes/Second-Brain-1/AI/Synth/evaluation/phase10/
├── quantitative_results.json
├── latent_ep1_pca.png
├── latent_ep5_pca.png
├── latent_final_pca.png
├── generated_maps/
│   ├── track1/
│   │   ├── generated_Easy.synth
│   │   ├── generated_Normal.synth
│   │   └── ...
│   └── ...
└── qualitative_assessment.md
```

Antigravity artifact namespace (for your own tracking):
```
/Users/marcus/.gemini/antigravity/brain/d3208ad5-9779-4922-83d7-473bbde04171/artifacts/phase10/
```

---

## 🚨 HANDOFF WARNINGS

1. **DO NOT restart training from scratch** — transformer_pilot_ep5.pt is valuable (hours of training)
2. **Training process may still be running** — check PID 9456 before any GPU-heavy operations
3. **Baseline model exists** at `models/checkpoints/best_model.pt` (Conv1D+LSTM) — use for comparison
4. **MPS memory** — Apple Silicon can OOM on large batches; current `BATCH_SIZE=8` with `ACCUM_STEPS=4` is stable

---

## 🔄 RELAY CHECKPOINT PROTOCOL

After completing Phase 10 tasks, update `task.md` like this:

```markdown
- [x] 1. Training Status Confirm
  - [x] Latest checkpoint: transformer_pilot_ep5.pt
  - [x] Converged/Running/Crashed: Converged (Finished Epoch 5)
- [x] 2. Quantitative Evaluation
  - [x] Recall@50ms: 53.61%
  - [x] Precision@50ms: 37.88%
  - [x] Exported to evaluation/phase10_quantitative.json
- [x] 3. Latent Space Verification
  - [x] PCA variance retained: 99.22% (with severe mathematical bounds overflows)
  - [x] Singularity status: SHATTERED (Chaotic Dispersion)
- [x] 4. Qualitative Generation
  - [x] Generated 15 beatmaps (3 tracks × 5 difficulties)
- [x] 5. Baseline Comparison
  - [x] Transformer vs Conv1D+LSTM results documented
```

---

## 📞 DECISION POINTS FOR MARCUS

At the end of Phase 10, Marcus needs to decide:

1. **Does recall@50ms exceed 70%?**
   - YES → Proceed to full-scale training (all 4,638 maps)
   - NO → Debug: data pipeline? architecture? more epochs?

2. **Is position accuracy usable (< 0.2 grid units error)?**
   - YES → Ready for human playtesting
   - NO → Need position head refinement

3. **Does Expert/Master difficulty generate coherent patterns?**
   - YES → Difficulty conditioning working
   - NO → May need difficulty-specific training or curriculum learning

4. **HerBB↔Antigravity Protocol** — confirm relay mechanism:
   - Continue using Kimi/Gemini via API?
   - Switch to local Hermes Agent for autonomous training?
   - Hybrid: antigravity for architecture, HerBB for orchestration?

---

## 🏁 SUCCESS CRITERIA FOR PHASE 10

Phase 10 is **COMPLETE** when:
- [ ] Latest checkpoint identified and evaluated
- [ ] Quantitative metrics captured in JSON
- [ ] ≥1 generated beatmap playable in SynthRiders (via Quest Link or editor)
- [ ] Qualitative assessment drafted
- [ ] `task.md` and `walkthrough.md` updated
- [ ] Relay prompt (this file) updated with actual results for next handoff

---

*Relay initiated by: HerBB*  
*Relay target: Antigravity*  
*Timestamp: 2026-04-27*  
*Session context preserved in: /Volumes/Second-Brain-1/AI/Synth/RELAY_PROMPT_PHASE_10.md*
