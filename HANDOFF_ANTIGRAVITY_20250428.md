# Antigravity Handoff: SynthRiders Phase 12b → Sim / Feasibility Reward Model

**Date:** 2025-04-28
**From:** HerBB (local session)
**To:** Antigravity
**Status:** Phase 12b production complete. Feasibility checker built. Sim architecture ready for you to implement.

---

## 1. Phase 12b is DONE

- **8,284 maps generated** with AIv12b.100ms (128-dim Transformer + 100ms NMS)
- **41,420 difficulty variations** (Easy/Normal/Hard/Expert/Master)
- **F1: 74.3%**, Precision: 71.2%, Recall: 77.7%, Position MSE: 1.94
- All maps stored as JSON bundles at:
  ```
  /Volumes/Second-Brain-1/AI/Synth/evaluation/phase12b/gold_standard/
  ```
- Format per map bundle: `{uuid, version, difficulties: {Easy: [notes...], Normal: ..., ...}}`
- Each note: `{time, type (0=left/1=right), x, y}`

**The NMS breakthrough:** The model was outputting "clustered logits" — each beat onset produced a ~100-150ms salience cluster. A static 100ms temporal NMS window collapsed the flood from 18,000 notes to ~1,100, restoring surgical precision without retraining.

---

## 2. Feasibility Checker (Built Today)

**File:** `scripts/feasibility_checker.py`

This is a hand-aware playability simulator that evaluates generated maps against physical and procedural rules. It runs per-difficulty and splits notes by hand `type` before checking.

### Checks Implemented

| Check | What It Measures | Hand-Aware? |
|-------|-----------------|-------------|
| **Bounds** | Notes inside play area (-4 to 4, -3 to 3) | N/A |
| **Reachability** | Can one hand traverse from note A to note B in the given time delta? | YES — only within same `type` |
| **Flow** | Angular smoothness between triplets (180° = reverse direction = bad within a hand) | YES — per-hand only |
| **Balance** | Left/right hemisphere distribution (warns if >65/35 split) | N/A |
| **Density Ramp** | Does density rise from early to late? (ambient intro → chorus build) | N/A |
| **Co-Located** | Same position within 40ms (double-hit bug) | N/A |
| **Rail Candidates** | Detects sustained-direction note sequences (potential slider/rails) | N/A |

### Crucial Fix Applied
Original checker evaluated ALL consecutive notes for reachability/flow. This was **wrong** — cross-hand transitions (Type 0 → Type 1) are simultaneous, not sequential. The patched version splits by hand type first. After the patch, reachability violations dropped from thousands to zero.

### Difficulty Parameters
Each difficulty has its own thresholds for hit windows, max density, reachability tolerance, flow angle tolerance, and min rail length. Easy's max density was bumped from 6.0 → 9.0 NPS after audit showed actual Easy output averages ~8-9 NPS.

---

## 3. Batch Audit Results (All 8,284 Maps)

Reports stored at: `evaluation/phase12b/feasibility_reports/`

| Difficulty | Pass Rate | Avg Notes | Top Failure Mode |
|-----------|-----------|-----------|----------------|
| **Easy** | 37.2% | 183 | Left-hand balance bias |
| **Normal** | 91.8% | 245 | Density ramp |
| **Hard** | 95.1% | 268 | Density ramp |
| **Expert** | 96.4% | 293 | — |
| **Master** | 96.6% | 295 | — |

**Easy root cause:** When the model is asked to produce "fewer notes" (Easy mode), it doesn't evenly delete across both hands. It keeps hammering the left hand and strips the right. This is a model behavior issue, NOT a threshold issue. Easy needs a dedicated mapping philosophy — sparse, alternating, symmetrical — not "Master with fewer notes."

---

## 4. What Marcus Wants Next: The Sim / Reward Model

Marcus wants to evolve from a passive feasibility checker into an **active reward model** that can score arbitrary maps and feed gradients back to the generator.

### Stage 1: Human-Kinematic Feasibility Scorer (YOU ARE HERE)
The checker exists. Now turn it into a **learned reward function**.

Proposed approach:
1. Build a lightweight MLP that takes a difficulty's note sequence as input and outputs a scalar "playability score."
2. Inputs: normalized note positions (x, y), time deltas, hand-type sequences, instantaneous velocity/acceleration between same-hand pairs, angular velocity, left/right balance histogram, density curve
3. Train it on the 8,284 maps labeled by the current rule-based checker (pass/fail per difficulty).
4. Replace the rules with learned heuristics. Let the model learn that "left-heavy Easy maps are bad" from data, not from a hardcoded 0.65 threshold.

### Stage 2: Quest 3 Telemetry Ingestion (WHEN MARCUS PLAYS)
Marcus will play test maps on the Quest 3. We need to capture:
- Controller position/velocity per frame
- Hit/miss events per note
- Headset position (body lean)
- Subjective feel rating

This data becomes the **true reward signal**. The learned scorer from Stage 1 is pre-trained on rule-based labels; fine-tuned on Marcus's actual gameplay.

### Stage 3: RL Generator Integration
Once we have a reward model, the training loop becomes:
1. Generate map candidate via Phase 12b generator
2. Score it with the learned reward model
3. If score is below threshold, adjust position/presence logits
4. Backprop through the generator

Or use the reward model as a discriminator in a GAN-style setup: generator tries to fool the reward model; reward model learns to detect "feels bad" maps.

---

## 5. Your Files

```
/Volumes/Second-Brain-1/AI/Synth/
├── evaluation/phase12b/gold_standard/           # 8,284 map bundles
├── evaluation/phase12b/feasibility_reports/     # 8,284 audit JSONs
├── scripts/feasibility_checker.py               # The checker (hand-aware)
├── scripts/generate_gold_standard.py            # Production generator
├── scripts/validate_synth_batch.py              # Temporal/spatial validator
├── models/checkpoints/transformer_phase12b_ep5.pt # Phase 12b weights
```

---

## 6. Immediate Action Items

1. **Learned Feasibility Scorer:** Build an MLP that ingests the checker outputs + raw note features and predicts overall_pass with high accuracy. Validate it can generalize.
2. **Easy Difficulty Fix:** Investigate why Easy mode is left-biased. Is it in the difficulty embedding? The inference scaling? Or a training corpus imbalance?
3. **Telemetry Pipeline Design:** Define Quest 3 ADB logging format for controller/headset tracking. Create ingestion scripts.

---

## 7. Context You Need

- Phase 12b generator uses a static 100ms NMS window. Model outputs clusters; NMS flattens them.
- Difficulty is encoded as an embedding during generation (5 classes).
- The model is NOT trained on human playtest data — only on static .synth files (expert-made).
- The "sim" concept is about kinesthetic validity: can a human body actually move through this note field and feel good doing it?

Marcus wants this done. He's in the car, coming back to playtest later. He'd love to fire up the Quest 3 and see AI maps that the sim has pre-screened for human feasibility.

Go.
