# Phase 10 Validation — Track Selection

## Selected Test Tracks

| BPM Range | Track | Artist | UUID | Why |
|-----------|-------|--------|------|-----|
| **Low (86)** | Pictures of You | The Cure | `da935995ce919f2e` | Classic ballad, slow, melodic — tests sustained note placement |
| **Mid (128)** | Warg | Nhato | `849ef40fc905105a` | EDM with build/drop structure — tests dynamic intensity shifts |
| **High (154)** | Da da dance | BabyMetal | `9be318efb916770f` | High-energy metal — tests rapid alternation & density |

All three have audio features pre-extracted and ready in `/dataset/audio_features/`.

---

## Execution Approved

Proceed with:
1. **Training status check** — confirm PID 9456 state and identify final checkpoint
2. **Create evaluation script** — `scripts/evaluate_checkpoint.py` with streaming loader
3. **Quantitative metrics** — Report Recall@50ms, Precision@50ms, Position MSE, Hand accuracy
4. **Latent re-viz** — Confirm singularity remains shattered
5. **Generate 15 maps** — 3 tracks × 5 difficulties each
6. **Baseline comparison** — Transformer vs Conv1D+LSTM

Export all artifacts to `/Volumes/Second-Brain-1/AI/Synth/evaluation/phase10/`.

Relay confirmed. Execute at will.

---

*Selection based on corpus scan of 50 representative maps, filtered for audio feature availability and BPM diversity across key genres (ballad, EDM, metal).*
