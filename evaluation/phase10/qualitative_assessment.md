# Phase 10 Validation: Transformer vs Baseline (Conv1D + LSTM)

## 1. Quantitative Breakdown
*   **Recall@50ms**: **53.61%** (Failed to breach >70% target)
*   **Precision@50ms**: **37.88%**
*   **Position MSE**: **19.8220** (Failed <0.2 target. Severe deviation)

### Baseline Comparison
The former Conv1D+LSTM (`best_model.pt`) architecture reliably sustained ~82% Recall with an MSE of ~0.15. The new `TransformerCausalDecoder` currently severely lags behind the previous network on localized precision, despite the Phase 9 Conv1D bridge. This is highly indicative of the Transformer struggling with the `tgt_t` autoregressive requirement cleanly during un-guided generation, causing error accumulation that blows out the coordinate matrix (evidenced by 19.82 MSE). 

## 2. Qualitative Beatmap Coherence
*   The **Difficulty Embeddings ARE working**. A clear scaling factor is visible across difficulties (e.g. `Da da dance` yields 1346 notes on Easy -> 1785 on Expert).
*   However, due to the high Position MSE, observing the matrices implies the notes are likely "spraying" rapidly rather than tracing smooth patterns. Subjectively, without the teacher-forcing mechanism, the network is likely relying entirely on the Conv1D-bridged audio features and wildly drifting across its positional targets.

## 3. Latent Space (Singularity) Status
*   Status: **SHATTERED (Anomalous)**
*   The raw zero-shot dispersion observed in Phase 9 maintained its chaos into Phase 10. `sklearn` PCA math continues to violently under/overflow from the sheer disjoint variance caused by the un-restricted `CausalConv1d` magnitudes. The grid is truly shattered, but it is currently un-aligned to a clean manifold.

## Final Verdict
We did not hit the >70% Recall / < 0.2 MSE bounds, and thus we **cannot** greenlight the 4,638 full convergence scale just yet. We need an architectural refinement—specifically addressing how the Decoder handles autoregressive targeting at inference compared to Teacher-Forced training.
