# Phase 11 Implementation Decision

## Critical Question: Transformer Initialization

**Question:** Fresh Kaiming uniform vs. load from transformer_pilot_ep5.pt?

## Decision: LOAD FROM EP5.PT (strict=False)

### Rationale

The Transformer's role has **shifted** but its **acoustic understanding** is valuable salvage:

| Component | Role in Phase 9 | Retain in Phase 11? | Why |
|-----------|-----------------|---------------------|-----|
| CausalConv1d bridge | Tactile response kernel | **YES** | Acoustic pattern memory (150ms window) |
| Cross-attention layers | Audio→Note mapping | **YES** | Learned rhythmic attention weights |
| Positional encoding | Temporal structure | **YES** | Universal, already converged |
| Output heads | Direct generation | **ADAPT** | Will shift to "correction" mode via training |
| coarse_proj (NEW) | LSTM memory integration | **Kaiming init** | Fresh, learns to read LSTM scaffold |

### The Teacher Forcing Problem

Phase 10's collapse was a **training procedure flaw**, not a **weight initialization flaw**:
- Bad: Always feeding ground truth history
- Good: The acoustic feature extraction (CausalConv1d → Attention) learned real rhythmic patterns

The 5 epochs weren't wasted — they built acoustic intuition. We're repurposing that intuition, not discarding it.

### Implementation Detail

```python
# In train_hybrid.py

import torch
from src.models.transformer import TransformerCausalDecoder
from src.models.baseline import BaselineBeatmapModel

# 1. Load LSTM (FROZEN)
baseline = BaselineBeatmapModel()
baseline.load_state_dict(torch.load("models/checkpoints/best_model.pt"))
baseline.requires_grad_(False)  # Frozen scaffold

# 2. Load Transformer (EP5.PT) with new coarse_proj
transformer = TransformerCausalDecoder(d_model=256, num_layers=4, d_audio=128, d_target=8)

# Load existing weights, ignore coarse_proj
state_dict = torch.load("models/checkpoints/transformer_pilot_ep5.pt")
transformer.load_state_dict(state_dict, strict=False)  # coarse_proj won't match, remains Kaiming init

# Train only: coarse_proj + minor finetune of decoder
# Higher LR for coarse_proj, lower LR for everything else
```

### Learning Rate Strategy

```python
optimizer = torch.optim.AdamW([
    {"params": transformer.coarse_proj.parameters(), "lr": 5e-4},      # NEW: aggressive learning
    {"params": [p for n, p in transformer.named_parameters() if "coarse_proj" not in n], 
     "lr": 1e-4}  # EXISTING: gentle finetuning
], weight_decay=0.01)
```

This discriminates: new integration layer learns fast, existing acoustic memory adapts slowly.

### Fallback Option

If Phase 11 metrics don't improve after 2 epochs:
```python
# Option: Re-initialize cross-attention layers too
# Keep only: CausalConv1d + PositionalEncoding
# Reset: All attention weights, output heads
```

But start with **full ep5.pt transfer** and diagnose from there.

---

## Execution Lock

**Marcus Decision:** Initialize Transformer from ep5.pt with strict=False, fresh Kaiming init only for coarse_proj.

**Rationale:** Preserve acoustic memory, learn integration from scratch.

**Learning Rate:** Discriminative (coarse_proj @ 5e-4, existing @ 1e-4).

**Ready for implementation.**
