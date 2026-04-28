# Claude Code Execution Prompt: SynthRiders AI Beatmap Generator
## Target Agent: Claude Code / kimi k2.6 / glm5.1
## Session Goal: Build Working Prototype in 1-2 Sessions
## Repo: https://github.com/marc0ne-s/synth-beatmap-gen

---

## Context

You are working on AI beatmap generation for SynthRiders (VR rhythm game). A 128-dim Transformer generates beatmaps from raw audio. The model was trained with MLE and produces dense maps (~15 NPS, 100 notes/frame). We're using RL to teach it sparse Easy mode (2.1 NPS).

**The core problem:** The model's presence logits are fully saturated (sigmoid ≈ 1.0 for all frames), so REINFORCE has zero variance (Advantage = 0.0). We bypass RL during density descent and use supervised BCE against a sparse oracle mask.

**Architecture:**
- Audio: 128-dim features (80 log-mel + 48 zero-pad during RL training)
- Model: Difficulty-conditioned causal Transformer (d=128, 8 layers, 8 heads)
- Output per frame: presence_logits (2 values for left/right hand)
- Inference: Apply 100ms NMS temporal peak detection post-hoc

---

## Task 1: Audit and Fix `train_rl_refinement.py`

### File Location
`scripts/train_rl_refinement.py` (check repo or create from scratch)

### Required Architecture

```python
# Difficulty targets (NPS = Notes Per Second)
DENSITY_TARGETS = [2.1, 3.5, 6.0, 9.0, 15.0]

# Reward component: exponential density bonus
# Target: 2.1 NPS for Easy. At 10 NPS: exp(-|10-2.1|/2.1) = exp(-3.76) ≈ 0.023
# At 2.1 NPS: exp(0) = 1.0

def compute_density_reward(nps, target_nps):
    return torch.exp(-torch.abs(nps - target_nps) / target_nps)

# Alignment: Fixed-K stratified oracle mask
# K_global = target_nps * duration_s
# e.g., 2.1 NPS × 4min = 2.1 × 240 = ~504 notes total for Easy
# Split evenly across track segments (e.g., 4 segments = ~126 per segment)
# Per segment: select top-K frames from REFERENCE MODEL's presence logits

def get_alignment_mask(ref_presence_logits, target_nps, duration_s, num_segments=4):
    """
    Create a binary mask of the K most musically salient frames.
    Stratified: divide track into segments, pick top-k per segment.
    
    ref_presence_logits: Tensor of shape (seq_len, 2) from reference model
    target_nps: float, e.g. 2.1 for Easy
    duration_s: total duration in seconds
    num_segments: number of segments to stratify across
    
    Returns: mask of shape (seq_len,) with EXACTLY K_global ones
    """
    seq_len = ref_presence_logits.shape[0]
    K_global = max(1, int(target_nps * duration_s))
    seg_len = seq_len // num_segments
    k_per_seg = max(1, K_global // num_segments)
    
    # Take max confidence across hands per frame
    ref_confidence = torch.sigmoid(ref_presence_logits).max(dim=-1).values
    
    mask = torch.zeros(seq_len)
    for i in range(num_segments):
        start = i * seg_len
        end = min((i + 1) * seg_len, seq_len)
        if end <= start:
            continue
        
        seg_conf = ref_confidence[start:end]
        _, top_indices = torch.topk(seg_conf, min(k_per_seg, end - start))
        mask[start:end][top_indices] = 1.0
    
    return mask

# Reward calculation per map sample
# BIAS_MAP: difficulty-indexed exploration bias (huge for low difficulties)
BIAS_MAP = {0: 2.0, 1: 1.0, 2: 0.5, 3: 0.0, 4: 0.0}

def compute_reward(position_pred, presence_logits, diff_idx, mask, ref_presence_logits, audio_duration):
    nps = presence_logits.sigmoid().sum().item() / audio_duration
    target_nps = DENSITY_TARGETS[diff_idx]
    
    # Density bonus: exponential penalty for deviation
    bonus = compute_density_reward(nps, target_nps)
    
    # Alignment: what fraction of output probability lands on oracle frames?
    probs = torch.sigmoid(presence_logits)
    alignment = (probs * mask).sum() / (probs.sum() + 1e-6)
    
    # Uniformity: prevent temporal clumping (ratio of min_density/max_density halves)
    # Not critical if fixed-K is working, but keep for safety
    
    # Playability score from scorer (only call scorer when NPS < 1.2× target)
    scorer_score = 0.0  # Will be filled by actual scorer inference
    
    # Apply Turbo Bypass: if density is way off, skip scorer entirely
    # Only use supervised BCE loss, ignore playability
    
    total_reward = (scorer_score + BIAS_MAP[diff_idx]) * bonus * alignment
    return total_reward, nps, bonus, alignment
```

### Turbo Bypass Implementation

```python
# TURBO BYPASS: When density is way off target, skip expensive scoring
# and use ONLY supervised BCE against the oracle mask.

SUPERVISED_WEIGHT = 20.0  # 20x multiplier for BCE term
BYPASS_THRESHOLD = 1.2   # If NPS > 1.2x target, bypass RL entirely

if current_nps > DENSITY_TARGETS[diff_idx] * BYPASS_THRESHOLD:
    # TURBO MODE: Pure supervised descent
    # Loss = w * BCE(logits, mask) — no REINFORCE, no scorer
    mask = get_alignment_mask(ref_logits, DENSITY_TARGETS[diff_idx], duration)
    sup_loss = F.binary_cross_entropy_with_logits(
        presence_logits, mask.to(presence_logits.device).float()
    )
    loss = SUPERVISED_WEIGHT * sup_loss
    # Skip REINFORCE completely — backprop only this supervised loss
else:
    # HANDOFF: Density is in the ballpark, switch to RL fine-tuning
    # Now compute: reward, advantage, REINFORCE gradient + KL penalty
    reward, nps, bonus, alignment = compute_reward(...)
    advantage = reward - baseline  # baseline = greedy (argmax) sample score
    pg_loss = -advantage * log_prob
    kl_loss = compute_kl(presence_logits, ref_presence_logits, beta_map[diff_idx])
    loss = pg_loss + kl_loss
```

### Critical Bugs to Fix

**Bug 1: The presence_logits shape must be (seq_len, 2) for left/right hands.**
If the model outputs flattened logits that combine both hands, the hand conditioning is broken.

**Bug 2: The ref_presence_logits must use the REFERENCE model, not the RL model.**
The ref model is Phase 12b MLE weights (frozen). Don't accidentally use current policy logits.

**Bug 3: The `F.binary_cross_entropy_with_logits` target shape must match logits exactly.**
If mask has shape (seq_len,) but logits are (seq_len, 2), you need to broadcast or reduce.
Recommended: `mask.unsqueeze(-1).expand_as(logits)` so both hands see the same oracle frame.

**Bug 4: NPS calculation must count across BOTH hands.**
```python
# WRONG: nps = sigmoid(logits).sum() / duration — counts frame predictions
# RIGHT: nps = sigmoid(logits).sum(dim=(0,1)) / duration — counts actual notes
nps = torch.sigmoid(presence_logits).sum().item() / duration
```

**Bug 5: The KL penalty must use per-Bernoulli BCE, not batch-wise KL divergence.**
Each frame's left/right presence is an independent Bernoulli, not a categorical.
```python
# WRONG: kl = F.kl_div(log_probs, ref_probs)
# RIGHT:
ref_sigmoid = torch.sigmoid(ref_presence_logits)
kl_loss = F.binary_cross_entropy_with_logits(presence_logits, ref_sigmoid) - \
          F.binary_cross_entropy_with_logits(ref_presence_logits, ref_sigmoid)
# Or simpler: just use the difference of logits against sigmoid targets
kl_loss = (F.binary_cross_entropy_with_logits(presence_logits, ref_sigmoid) - 
           F.binary_cross_entropy_with_logits(ref_presence_logits, ref_sigmoid)).mean()
```

### Difficulty-Aware Hyperparameters

```python
BETA_MAP = {0: 0.0, 1: 0.05, 2: 0.1, 3: 0.4, 4: 0.7}
BIAS_MAP = {0: 2.0, 1: 1.0, 2: 0.5, 3: 0.0, 4: 0.0}
PG_WEIGHT_MAP = {0: 5.0, 1: 1.0, 2: 1.0, 3: 2.0, 4: 2.0}
DENSITY_TARGETS = [2.1, 3.5, 6.0, 9.0, 15.0]
EXPLORE_BIAS = 0.1  # Added to Score during early RL epochs before Scorer awakens
HARD_FLOOR_NPS = 1.0  # Any map below this gets zero reward through density penalty
```

### Training Loop Skeleton

```python
for epoch in range(max_epochs):
    for track_idx, track_data in enumerate(dataloader):
        audio = track_data["audio"]
        duration = track_data["duration"]
        diff_idx = random.randint(0, 4)  # Contrastive sampling
        
        # Forward pass
        with torch.no_grad():
            ref_logits = reference_model(audio, diff_idx=diff_idx)
        
        current_logits = policy_model(audio, diff_idx=diff_idx)
        
        # Compute oracle mask from REFERENCE logits
        mask = get_alignment_mask(ref_logits, DENSITY_TARGETS[diff_idx], duration)
        
        # Compute current NPS
        probs = torch.sigmoid(current_logits)
        current_nps = probs.sum().item() / duration
        
        if current_nps > DENSITY_TARGETS[diff_idx] * BYPASS_THRESHOLD:
            # === TURBO BYPASS MODE ===
            sup_loss = F.binary_cross_entropy_with_logits(
                current_logits, mask.unsqueeze(-1).expand_as(current_logits).float()
            )
            loss = SUPERVISED_WEIGHT * sup_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
            optimizer.step()
            
        else:
            # === RL FINE-TUNING MODE ===
            # Sample from policy (not greedy)
            sampled_presence = torch.bernoulli(probs)
            log_prob = ...  # compute log_prob of the sample
            
            # Run through scorer + reward computation
            scorer_input = extract_features_from_sample(sampled_presence, audio)
            scorer_score = scorer_model(scorer_input)
            
            reward = (scorer_score + BIAS_MAP[diff_idx]) * compute_density_reward(current_nps, DENSITY_TARGETS[diff_idx])
            
            # Baseline = greedy (argmax) score
            baseline_score = scorer_model(extract_features_from_greedy(probs, audio))
            advantage = reward - baseline_score
            
            pg_loss = -advantage * log_prob
            
            # KL penalty (except for Easy/Normal where beta=0)
            if BETA_MAP[diff_idx] > 0:
                ref_sigmoid = torch.sigmoid(ref_logits)
                kl_loss = (F.binary_cross_entropy_with_logits(current_logits, ref_sigmoid) - 
                          F.binary_cross_entropy_with_logits(ref_logits, ref_sigmoid)).mean()
                kl_loss = BETA_MAP[diff_idx] * kl_loss
            else:
                kl_loss = 0.0
            
            loss = PG_WEIGHT_MAP[diff_idx] * pg_loss + kl_loss
            
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Logging
        print(f"Epoch {epoch} Track {track_idx} Diff={diff_idx} "
              f"NPS={current_nps:.2f} Loss={loss.item():.4f} "
              f"Mode={'TURBO' if current_nps > DENSITY_TARGETS[diff_idx]*BYPASS_THRESHOLD else 'RL'}")
```

### Expected Output

After fixing the Turbo Bypass, expect:
- Track 1-10: Easy Diff=0 still at ~10-15 NPS
- Track 20-50: Easy starts dropping to 8-10 NPS
- Track 50-100: Easy hits 3-5 NPS
- Track 100+: Easy stabilizes at 2.1±0.5 NPS
- Master Diff=4 should stay at 14-16 NPS throughout

If Easy is still at 10 NPS after 100 tracks: check that the supervised loss is actually backpropagating to the presence head (print gradients on `presence_logits` to confirm non-zero).

---

## Task 2: Build End-to-End Inference Pipeline

Create `scripts/inference_pipeline.py` that does:

1. **Load audio file** (WAV/MP3) via librosa
2. **Extract 128-dim features** (80 log-mel + 48 pad, or full 128 if available)
3. **Load Phase 12b model** + optional RL weights if available
4. **Generate 5 difficulties** via autoregressive sampling
5. **Apply 100ms NMS** temporal peak detection post-hoc
6. **Format output** as .synth beatmap files
7. **Run feasibility checker** on generated maps

### Output Format (.synth)
```json
{
  "trackId": "audio_filename",
  "difficulty": 3,  // 0=Easy, 4=Master
  "notes": [
    {"time": 1.234, "x": -0.5, "y": 0.8, "z": 2.0, "hand": "left", "type": "note"},
    ...
  ]
}
```

### CLI Interface
```bash
python scripts/inference_pipeline.py \
    --audio input.mp3 \
    --difficulties all \
    --output-dir maps/ \
    --checkpoint models/checkpoints/transformer_phase12b_ep5.pt \
    --scorer models/checkpoints/scorer_v0.pt
```

---

## Task 3: Create Feasibility Audit Script

Build `scripts/run_feasibility_audit.py` that evaluates a batch of generated maps across all 5 difficulties.

**Input:** Directory of generated .synth files or JSON outputs
**Output:** `audit_report.json` with:

```json
{
  "timestamp": "2026-04-28T10:00:00Z",
  "total_maps": 100,
  "difficulty_breakdown": {
    "easy": {
      "pass_rate": 0.42,
      "avg_notes": 165,
      "avg_nps": 2.3,
      "failures": ["imbalance", "too_dense"]
    },
    "normal": {...},
    "hard": {...},
    "expert": {...},
    "master": {...}
  },
  "overall_pass_rate": 0.78,
  "recommendation": "Easy needs more reduction; Master is solid."
}
```

**Failure modes to track:**
- Hand imbalance (σ > 0.15 between left/right count)
- Temporal clumping (density ratio < 0.33 between halves)
- Density mismatch (NPS outside target range)
- Playability score < 0.5
- NPS < 1.0 (sparse trough)
- NPS > 1.5× target (overshoot)

---

## File Checklist

After this session, these files should exist and work:

| File | Status |
|------|--------|
| `scripts/train_rl_refinement.py` | ✅ Fixed Turbo Bypass runs without crash |
| `scripts/inference_pipeline.py` | ⬜ End-to-end audio → .synth generator |
| `scripts/run_feasibility_audit.py` | ⬜ 5-difficulty audit with JSON report |
| `scripts/visualize_map.py` | ⬜ Optional: 3D plot/note chart viewer |
| `models/checkpoints/rl_phase12b.pt` | ⬜ Saved RL weights (or report why not yet) |

---

## Hard Constraints

1. **MPS-compatible** (Apple Silicon, no CUDA)
2. **Audio processing must use librosa** (standard)
3. **Zero external dependencies beyond:** torch, librosa, numpy, json, tqdm
4. **Scorer v0 weights frozen** (no co-training, prevent reward hacking)
5. **Phase 12b reference model frozen** (only gate with KL, don't retrain)
6. **No hallucinations** — if a model checkpoint doesn't exist, report it, don't fake it

## First Action

1. Clone the repo: `git clone https://github.com/marc0ne-s/synth-beatmap-gen.git`
2. Read `scripts/train_rl_refinement.py` — if it exists, audit it
3. If it doesn't exist, build it from this prompt
4. Run a single-track test: `python scripts/train_rl_refinement.py --test-mode`
5. Report what crashes, what works, what NPS numbers you see

## Success Criteria

A working prototype means:
- ✅ Script runs end-to-end without crash
- ✅ Easy mode generates maps at 2.1±1.0 NPS
- ✅ Master mode stays at 12+ NPS
- ✅ Feasibility audit shows ≥40% Easy pass rate
- ✅ Inference pipeline works on a new audio file
- ✅ Maps are output in correct .synth format

This is a research prototype, not a production app. Bugs are expected. Report them honestly.
