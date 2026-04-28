# SYNTHRIDERS AI — RELAY PROMPT: PHASE 11
# THE AUTOREGRESSIVE FIX — FROM TEACHER FORCING TO ITERATIVE REFINEMENT

> **Context:** Phase 10 COMPLETE — Diagnostic failure revealed core architectural flaw  
> **Previous State:** Phase 9 Hybrid Transformer (Conv1D bridge) converged  
> **Critical Finding:** Teacher forcing collapse — model depends on ground truth history  <br>
> **Mission:** Fix autoregressive inference logic so model generates from its own predictions

---

## 🚨 PHASE 10 POST-MORTEM: WHAT WENT WRONG

### The Numbers (Brutal Truth)

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Recall@50ms | >70% | **53.61%** | ❌ FAIL |
| Precision@50ms | >50% | **37.88%** | ❌ FAIL |
| Position MSE | <0.2 | **19.82** | ❌ CATASTROPHIC |

### Root Cause Analysis: Teacher Forcing Collapse

**The Problem:**
The TransformerCausalDecoder uses **teacher forcing** during training:

```python
# Forward pass (TRAINING)
shifted_targets = torch.zeros_like(target_features)
shifted_targets[:, 1:, :] = target_features[:, :-1, :]  # <-- GROUND TRUTH history
v_tgt = self.target_proj(shifted_targets)  # <-- Always perfect history
```

During **inference**, there's no ground truth. The model receives:
```python
# Forward pass (INFERENCE — Generation Mode)
prev_pred = model.predict_step(...)  # <-- Its OWN noisy prediction
shifted_targets[:, t, :] = prev_pred   # <-- Error compounds
```

**Result:** Spatial drift explodes because:
1. Training distribution = perfect history + noise
2. Inference distribution = accumulated error + noise
3. Mismatch causes compounding spatial drift (MSE 19.82 vs target 0.2)

**The Conv1D+LSTM baseline survived** because recurrent models are less sensitive to this — they naturally smooth over historical noise. Transformers are brutally literal.

---

## 🎯 PHASE 11 OBJECTIVE

Fix the autoregressive inference gap. Three viable approaches:

### Option A: Scheduled Sampling (Gradual Transition)
During training, progressively replace ground truth with model's own predictions:

```python
# Teacher forcing ratio decays over epochs
if random.random() < teacher_forcing_ratio:
    decoder_input = ground_truth[t-1]
else:
    decoder_input = model_prediction[t-1]  # Model eats its own cooking
```

**Pros:** Minimal code change, proven technique
**Cons:** Slow convergence, still suffers from exposure bias

### Option B: Iterative Refinement (Multi-Pass Denoising)
Generate an initial "noisy" draft, then refine it via residual corrections:

```python
# Pass 1: Draft generation (fast, potentially messy)
draft = model.generate_draft(audio_features)

# Pass 2: Refinement pass (model corrects its own errors)
refined = model.refine(draft, audio_features)

# Optional Pass 3+: Further refinement
```

**Pros:** Parallelizable refinement, matches diffusion model success
**Cons:** Computationally heavier, needs refinement head

### Option C: Hybrid Autoregressive + Non-Autoregressive
Keep Conv1D+LSTM for coarse generation, use Transformer for refinement:

```python
# Stage 1: Conv1D+LSTM generates rough beatmap (robust to noise)
coarse_map = lstm_generator(audio)

# Stage 2: Transformer refines positions/velocities
detailed_map = transformer_refiner(coarse_map, audio)
```

**Pros:** Leverages proven LSTM robustness + Transformer precision
**Cons:** Two models to train/maintain

---

## 📋 PHASE 11 IMPLEMENTATION PLAN

### Task 1: Architecture Decision — LOCKED: OPTION C (Hybrid Cascade)

**✅ DECISION MADE by Marcus:** Option C — Hybrid Cascade

**Rationale:** Leverages proven Conv1D+LSTM (`best_model.pt`, 57% recall) as coarse generator. Transformer focuses exclusively on refinement/correction. Lowest risk, highest confidence, fastest path to playable maps.

### Task 2: Implement Selected Fix

#### If Option A (Scheduled Sampling):
1. Modify `TransformerCausalDecoder.forward()` to accept `teacher_forcing_ratio` parameter
2. Create training loop that anneals ratio: `1.0 → 0.5 → 0.3 → 0.1` over epochs
3. During low-ratio periods, model uses `argmax/sample` from its own logits

#### If Option B (Iterative Refinement):
1. Add `RefinementHead` to `TransformerCausalDecoder`:
   - Takes `draft_predictions` as additional input
   - Outputs `delta_positions` (residual corrections)
2. Modify loss: `L_draft + L_refinement`
3. Generate draft in forward pass, refine it, compare to GT

#### If Option C (Hybrid Cascade):
1. Load `models/checkpoints/best_model.pt` (Conv1D+LSTM)
2. Freeze LSTM, use as "coarse generator"
3. Modify Transformer to accept coarse map as additional memory
4. Train only Transformer refinement layers

### Task 3: Re-Evaluation Protocol

After fix implementation:

```bash
# Re-run same evaluation
python scripts/evaluate_checkpoint.py \
  --checkpoint models/checkpoints/transformer_pilot_v2_ep5.pt \
  --test-tracks da935995ce919f2e,849ef40fc905105a,9be318efb916770f
```

**Target for Phase 11:**
- Recall@50ms: >65% (progress toward 70%)
- Position MSE: <1.0 (two orders of magnitude better)
- Qualitative: Generated maps must be **playable** (even if imperfect)

### Task 4: Playability Test (NEW!)

Load generated `.synth` files into SynthRiders Editor and verify:
- [ ] Notes appear in expected timing windows
- [ ] No "impossible" jumps (notes requiring 2+ meter arm extension)
- [ ] Hand alternation feels natural (not all left or all right)

---

## 🔧 TECHNICAL SPECIFICATION

### Current Broken Code (in `transformer.py` line 124-125)
```python
shifted_targets = torch.zeros_like(target_features)
shifted_targets[:, 1:, :] = target_features[:, :-1, :]
```

### Option A Fix (Scheduled Sampling)
```python
def forward(self, audio_features, target_features, difficulty_idx, 
            teacher_forcing_ratio=1.0):
    B, T, _ = audio_features.shape
    
    # Initialize with zeros or start token
    decoder_input = torch.zeros_like(target_features[:, 0:1, :])
    outputs = []
    
    for t in range(T):
        # Project current input
        v_audio = self.audio_proj(audio_features[:, t:t+1, :])
        v_tgt = self.target_proj(decoder_input)
        
        # [rest of transformer pass...]
        pred = self.output_head(x)
        outputs.append(pred)
        
        # Teacher forcing vs. autoregressive
        if random.random() < teacher_forcing_ratio:
            decoder_input = target_features[:, t:t+1, :]  # Ground truth
        else:
            decoder_input = pred.detach()  # Model's own prediction
    
    return torch.cat(outputs, dim=1)
```

### Option B Fix (Iterative Refinement Draft)
```python
class TransformerCausalDecoder(nn.Module):
    def __init__(self, ...):
        # ... existing layers
        self.refinement_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # Concatenate draft + audio
            nn.GELU(),
            nn.Linear(d_model, d_target)       # Residual corrections
        )
    
    def forward(self, audio_features, target_features=None, 
                mode='train', num_passes=1):
        
        if mode == 'train':
            # Standard teacher forcing for draft
            draft = self.generate_draft(audio_features, target_features)
            
            # Refinement target = GT - Draft
            residual_target = target_features - draft
            
            # Predict residual
            residual_pred = self.refinement_head(
                torch.cat([draft, audio_features], dim=-1)
            )
            
            return draft + residual_pred
        
        elif mode == 'inference':
            # Iterative passes
            current = self.generate_draft(audio_features)
            
            for _ in range(num_passes):
                corrections = self.refinement_head(
                    torch.cat([current, audio_features], dim=-1)
                )
                current = current + corrections * 0.5  # Damped update
            
            return current
```

### Option C Fix (Hybrid Cascade)
```python
class HybridGenerator:
    def __init__(self):
        self.lstm = load_baseline_model("best_model.pt")
        self.lstm.eval()  # Frozen
        
        self.transformer = TransformerCausalDecoder(...)
        # Transformer takes coarse LSTM output as additional input
    
    def forward(self, audio_features, target_features):
        with torch.no_grad():
            coarse = self.lstm(audio_features)
        
        # Transformer refines coarse LSTM output
        refined = self.transformer(
            audio_features, 
            target_features,
            coarse_memory=coarse  # New parameter
        )
        return refined
```

---

## 📊 DECISION MATRIX

| Approach | Implementation Time | Expected Recall Gain | Computation Cost | Risk |
|----------|--------------------|--------------------|----------------|------|
| A: Scheduled Sampling | 2-4 hours | +10-15% | Same | Low |
| B: Iterative Refinement | 1-2 days | +15-25% | 2-3x inference | Medium |
| C: Hybrid Cascade | 4-6 hours | +20-30% | 1.5x inference | Low-Medium |

---

## 🏁 PHASE 11 COMPLETION CRITERIA

Phase 11 is **COMPLETE** when:
- [ ] Architecture fix implemented (A, B, or C)
- [ ] Model re-trained for ≥3 epochs
- [ ] **Recall@50ms ≥ 65%** (progress toward 70%)
- [ ] **Position MSE ≤ 1.0** (from catastrophic 19.82)
- [ ] Qualitative: Generated maps are **human-playable**
- [ ] Validation via SynthRiders Editor (or at least visual inspection)

---

## 🎮 OPTIONAL: SYNTHETIC AUDIO SANITY CHECK

Before training on real audio, verify the autoregressive fix works on synthetic data:

```bash
python scripts/quick_test_autoregressive.py \
  --model transformer_v2 \
  --synthetic-audio \
  --teacher-forcing-ratio 0.0  # Full autoregressive mode
```

If it can't even track synthetic beats without teacher forcing, the fix is insufficient.

---

## 🔀 HERBB → ANTIGRAVITY HANDOFF

**HerBB notes:**
- Phase 10 was a diagnostic success masquerading as failure
- We now know the exact failure mode (teacher forcing → inference gap)
- The architecture is sound; the training procedure is wrong
- Conv1D+LSTM baseline remains viable fallback

**Antigravity mission:**
1. Review Options A/B/C above
2. Implement Marcus-selected fix
3. Re-train and re-evaluate
4. Report new Recall/MSE metrics
5. Generate and validate playable beatmaps

**Relay files updated:**
- `RELAY_PROMPT_PHASE_10.md` — Phase 10 results archived
- `RELAY_PROMPT_PHASE_11.md` — This file
- `evaluation/phase10/qualitative_assessment.md` — Detailed failure analysis

---

*The Singularity is shattered. Now we teach the model to walk without crutches.*
