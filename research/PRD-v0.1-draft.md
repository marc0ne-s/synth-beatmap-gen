---
title: SynthRiders AI Beatmap Generator — Product Requirements Document
author: Marcus + HerBB
version: 0.1-draft
status: Draft — Pending Review
date: 2026-04-28
---

# SynthRiders AI Beatmap Generator
## Product Requirements Document (PRD)

---

## 1. Executive Summary

SynthRiders AI Beatmap Generator is an intelligent music-to-game converter that takes any audio file and automatically generates playable, difficulty-calibrated beatmaps for the VR rhythm game **SynthRiders**.

The product transforms raw audio into five difficulty tiers (Easy, Normal, Hard, Expert, Master) using a learned playability model trained on human-created maps. Unlike procedural or rule-based beatmap generators, this system learns the *feel* of a good map — where to place notes, which hand to use, and how to maintain kinetic flow — from thousands of human-designed examples.

**Primary Goal:** Reduce map creation time from 4-8 hours of manual beat-mapping to under 60 seconds of automated generation, while maintaining quality that passes both algorithmic playability validation and human playtest approval.

---

## 2. Problem Statement

### 2.1 The Core Pain
SynthRiders, like all rhythm games, depends entirely on beatmaps. Without maps, the game is unplayable. Yet creating a good beatmap is an art form requiring:
- Deep musical intuition (hearing beats, anticipating drops, recognising phrases)
- Spatial design sense (where notes go in 3D space, hand balance)
- Hours of iteration and playtesting
- Skill in the editor software

The result: most players have access to maps for only a few hundred popular tracks. The long tail of music — personal libraries, underground artists, unreleased tracks — has zero maps.

### 2.2 Existing Solutions & Why They Fail

| Approach | Why It Fails |
|----------|-------------|
| Human mappers | 4-8 hours per track, bottlenecked by skilled labour |
| Beat saber auto-mappers | Don't understand SynthRiders' unique mechanics (3D orbs, special notes, hand-specific streams) |
| Simple onsets-to-notes | Notes land on beats but feel robotic, unplayable, unmusical |
| Template-based | One-size-fits-all; loses song identity at higher difficulties |

### 2.3 Our Insight
The model doesn't need to *understand* music like a human. It needs to learn the correlation between audio features and the human-designated placements that feel good. With ~8,861 human maps and a learned playability model scoring AUROC 0.9988, the AI can out-produce human intuition at scale.

---

## 3. Target Users

### 3.1 Primary: SynthRiders Content Creators
- Streamers and YouTubers who need maps for their background music
- Players who want to play their personal music library
- Beatmap curators building packs or playlists

**Needs:** Speed, quality, batch processing, and maps that feel "right."

### 3.2 Secondary: Casual SynthRiders Players
- Want to play any song they like
- Don't know how to use editors, don't want to learn
- Willing to accept "good enough" maps for the convenience

**Needs:** One-click generation, no technical setup, confidence the map won't suck.

### 3.3 Tertiary: Beatmap Artists (Human Mappers)
- Professionals who want a fast starting point
- Use the AI draft as a scaffold, then refine by hand

**Needs:** Editable output, precise control, and the ability to override AI decisions.

---

## 4. Product Vision

> **"Drop any song. Get five good maps. Play in VR within 60 seconds."**

The product is not a beatmap editor. It is a **music-to-map converter**. The interface should feel like a video transcoder — simple input, clear output, quality indicators — not a DAW or professional creative tool.

### 4.1 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Generation time (5 difficulties) | < 60s per track | Timing script |
| Expert/Master pass rate | ≥ 85% | Feasibility scorer |
| Easy pass rate | ≥ 40% (targeting ≥ 60%) | Feasibility scorer |
| Expert/Master "feels human" | ≥ 70% player approval | Blind test (future) |
| Batch processing | 100 tracks unattended | Long-running test |
| Time-to-first-map | < 30s from launch | Manual timing |

---

## 5. Core Features

### 5.1 MVP (Phase 12b + RL)

| Feature | Description | Status |
|---------|-------------|--------|
| **Audio Ingest** | Drop WAV/MP3, auto-extracts 128-dim features | ✅ |
| **5-Difficulty Generation** | Easy, Normal, Hard, Expert, Master | ✅ |
| **Playability Validation** | Runs scorer + rule-based checker per map | ✅ Scorer v0 |
| **Batch Processing** | Process 100+ tracks in queue | ⬜ |
| **.synth Output** | SynthRiders-compatible beatmap format | ✅ |
| **Feasibility Audit Report** | JSON summary with pass/fail per difficulty | ⬜ |
| **Temporal NMS Post-Processing** | Collapses clustered logits into clean notes | ✅ |

### 5.2 v2 (Post-Prototype)

| Feature | Description |
|---------|-------------|
| **Map Editor** | Cross-platform timeline editor (Electron/Tauri) with 3D preview |
| **Streaming Integration** | Paste Spotify/Apple Music links → auto-fetch + generate |
| **Style Profiles** | Learn mapper "style" from user's favourite maps |
| **Quest 3 Telemetry** | VR playtest capture for Stage 2 playability validation |
| **Community Sharing** | Publish maps, rate others, upvote/downvote |
| **Real-Time Preview** | Before saving, scrub through the map in 3D |

### 5.3 Will Not Do (Anti-Features)

Explicitly out of scope to prevent scope creep:
- **Audio synthesis or modification** (we read audio, we don't create it)
- **Procedural music generation** (maps for existing songs only)
- **Video or lighting effects** (SynthRiders' visual environment is separate)
- **Multiplayer or competitive features** (game-side, not generator-side)
- **Beatmap for non-SynthRiders games** (at least not in v1)

---

## 6. Technical Architecture

### 6.1 Data Flow

```
Audio Input (WAV/MP3)
    → librosa feature extraction (80 log-mel + 12 chroma + 36 onset/peak)
    → 128-dim audio feature tensor
    → Difficulty-conditioned Transformer Generator
    → Raw presence logits per frame × 2 (left/right)
    → Temporal NMS (100ms window) → collapses clusters
    → Position regression + hand classification
    → .synth beatmap file (JSON)
    → Feasibility Scorer validation
    → Audit report (pass/fail per difficulty)
```

### 6.2 Models

| Model | Purpose | Size | Status |
|-------|---------|------|--------|
| **Phase 12b Transformer** | Generator: audio → beatmap | ~15MB | ✅ Trained |
| **Playability Scorer v0** | Binary classifier: playable/unplayable | 2.3MB | ✅ Trained (AUROC 0.9988) |
| **RL Refinement (Stage 3)** | Fine-tunes generator using scorer as reward | Same as Phase 12b | 🔄 ACTIVE (Run 3.15) |

### 6.3 Hardware Target

- **Primary:** macOS (Apple Silicon M4 Pro) — Marcus's dev machine
- **Secondary:** Cross-platform (Linux, Windows) via PyTorch
- **MPS-compatible:** No CUDA dependency
- **Minimum:** 16 GB RAM, Apple Silicon or modern x86 GPU
- **Recommended:** 32 GB RAM, fast SSD for batch processing

### 6.4 Key Technical Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Audio hop length | 10ms (100 fps) | Industry standard for rhythm alignment |
| NMS window | 100ms | Collapses 15-frame clusters into single notes |
| Max sequence length | 24,000 frames (~4 minutes) | Covers 99% of pop/EDM tracks |
| Transformer dims | 128 | Balance between capacity and speed |
| Attention heads | 8 | Captures multiple beat scales |
| Layers | 8 | Sufficient for musical phrase structure |

---

## 7. User Flow

### 7.1 MVP CLI Flow

```bash
# Step 1: Drop audio
python generate.py --audio "my_track.mp3" --output ./maps/

# Step 2: Wait ~60 seconds
[INFO] Extracting audio features...
[INFO] Generating 5 difficulties...
[INFO] Running feasibility scorer...
[INFO] Auditing maps...

# Step 3: Get audit report
[INFO] Audit complete:
  Easy:     PASS (2.3 NPS, 0.42 playability)
  Normal:   PASS (3.8 NPS, 0.58 playability)
  Hard:     PASS (6.2 NPS, 0.72 playability)
  Expert:   PASS (11.7 NPS, 0.89 playability)
  Master:   PASS (13.4 NPS, 0.94 playability)

# Step 4: Play in SynthRiders
cp ./maps/my_track_*.synth ~/SynthRiders/CustomSongs/
```

### 7.2 Batch Flow

```bash
python batch_generate.py --input-dir ./album/ --output-dir ./maps/ --parallel 4
# Processes all tracks in folder, writes audit report per album
```

### 7.3 Future GUI Flow (v2)

1. User drags audio file(s) into app window
2. Progress bar shows "Analysing... Generating... Auditing..."
3. List view shows each difficulty with: pass/fail, note count, predicted playability
4. User clicks difficulty to preview (3D timeline scrub)
5. User clicks "Export" to save .synth file
6. Optional: "Open in Editor" to refine by hand

---

## 8. Playability Model Design

### 8.1 Scorer v0 (Current)

**Model:** Bi-LSTM + MLP classifier
**Input features:**
- Hand balance (left vs right note count, σ)
- NPS (notes per second)
- Wrist angle variance (how awkward are positions?)
- Reach distance (how far from centre?)
- Temporal symmetry (are patterns mirrored across hands?)
- Onset alignment (do notes land on musical beats?)

**Output:** P(playable) ∈ [0, 1]
**AUROC:** 0.9988 on holdout (very strong discriminator)
**Key weakness:** Trained on Phase 12b output distribution, so it's blind to genuinely novel map types.

### 8.2 Scorer v1 (Future — Stage 2)

**Data source:** Quest 3 VR telemetry
**Additional features:**
- Actual player movement trajectories
- Miss rates per note pattern
- Hand fatigue (measured from controller velocity variance)
- Head movement (did the player look ahead / anticipate?)
- Flow state indicators (smooth, continuous motion)

**Why this matters:** The current scorer predicts "does this look like a human map?" not "is this actually fun to play?" VR telemetry tells us the latter.

---

## 9. Quality Standards

### 9.1 Algorithmic Validation (Automated)

Every generated map must pass:

| Gate | Easy | Normal | Hard | Expert | Master |
|------|------|--------|------|--------|--------|
| Pass rate target | ≥ 40% | ≥ 60% | ≥ 75% | ≥ 85% | ≥ 90% |
| NPS target | 2.1±1.0 | 3.5±1.0 | 6.0±1.5 | 9.0±2.0 | 15.0±3.0 |
| Hand balance σ | ≤ 0.15 | ≤ 0.15 | ≤ 0.12 | ≤ 0.10 | ≤ 0.08 |
| Temporal uniformity ratio | ≥ 0.33 | ≥ 0.33 | ≥ 0.40 | ≥ 0.50 | ≥ 0.50 |
| Onset-beat correlation | ≥ 0.30 | ≥ 0.40 | ≥ 0.50 | ≥ 0.60 | ≥ 0.70 |
| Playability score | ≥ 0.30 | ≥ 0.45 | ≥ 0.60 | ≥ 0.80 | ≥ 0.90 |

### 9.2 Human Validation (Future)

Blind A/B test with SynthRiders players:
- Can they distinguish AI-generated from human maps?
- Which do they rate higher for "fun" and "flow"?
- Target: ≥ 70% "can't tell" or "AI feels better" for Expert/Master

---

## 10. Design Principles (UI/UX)

Marcus's explicit preferences — non-negotiable:

| Principle | Implementation |
|-----------|---------------|
| **Light, warm, organic** | Paper-texture, watercolor, slow-drift backgrounds |
| **Not dark, not geometric** | No concentric/orbital layouts, no neon-on-black |
| **Friction removal** | Every extra click is a failure. Drag, wait, done. |
| **Australian female TTS** | Brook Johnson (AU) for all audio briefings |
| **Naming precision** | Exact terminology matters. "Phase 12b" not "the model." |
| **Show progress, not options** | Default to sensible settings, expose controls only on demand |
| **Music comes first** | The audio should be audible/previewable at all times |

---

## 11. Roadmap

### Phase 1: Data Ingestion (COMPLETE)
- Ingest 8,861 human maps
- Parse .synth format, extract audio features
- Build clean training corpus
- **Deliverable:** Clean dataset, feature pipeline

### Phase 2: Model Training (COMPLETE)
- Train Transformer on human maps (MLE)
- Deploy Temporal NMS (100ms) for precision recovery
- Achieve 71% precision, 77.7% recall
- **Deliverable:** Phase 12b checkpoint

### Phase 2.5: Playability Model (COMPLETE)
- Extract playability features from human maps
- Train Bi-LSTM + MLP scorer
- Achieve AUROC 0.9988
- **Deliverable:** Scorer v0 checkpoint

### Phase 3: RL Refinement (ACTIVE)
- Use Scorer v0 as reward signal
- Fix Easy mode (0.32 → 0.80 target)
- Contrastive multi-difficulty fine-tuning
- **Deliverable:** RL-refined generator, Phase 12b.Final Audit passes

### Stage 2: VR Telemetry (FUTURE)
- Collect Quest 3 controller + headset data
- Build Scorer v1 with actual playability signal
- Validate against human playtesters

### Stage 3: Standalone App (FUTURE)
- Electron or Tauri cross-platform application
- Audio drag-and-drop → .synth export
- Batch processing queue
- Minimal, warm, non-intrusive UI

### Stage 4: Editor Integration (FUTURE)
- Timeline-based map editor
- 3D preview with VR viewport
- AI-generated draft + human refinement
- Community sharing platform

### Stage 5: Streaming (FUTURE)
- Paste Spotify/Apple Music link
- Auto-fetch metadata + generate map
- Streaming audio (metadata only, no raw audio access)

---

## 12. Dependencies & Risks

### 12.1 Technical Dependencies

| Dependency | Version | Source | Risk |
|------------|---------|--------|------|
| PyTorch | ≥ 2.0 | pip | MPS stability on Apple Silicon |
| librosa | ≥ 0.10 | pip | Audio feature extraction speed |
| NumPy | ≥ 1.24 | pip | None |
| tqdm | ≥ 4.65 | pip | None |
| SynthRiders game | Latest | External | Map format changes |

### 12.2 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| MPS memory crash during batch | Medium | High | Add periodic `torch.mps.empty_cache()`, chunk processing |
| Easy mode never hits 2.1 NPS | Medium | High | If RL fails, fall back to post-hoc density scaling (heuristic pruning) |
| Scorer v0 reward hacking | Low | High | Freeze scorer weights, monitor for co-adaptation |
| Real maps distribution ≠ training | High | Medium | Ingest 4,000 real user maps ASAP (already planned) |
| Map format breaking change | Low | High | Abstract map I/O layer, version-check on load |

### 12.3 Business Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Human mappers view AI as threat | High | Medium | Frame as "draft assistant" not "replacement"; support editor export |
| Copyright issues (auto-generating maps) | Medium | High | Generate only for user-provided audio; no redistribution of generated maps |
| Performance on non-EDM genres | Medium | Medium | Start with EDM/pop focus; expand to rock/hip-hop later |

---

## 13. Open Questions (For Discussion)

1. **Commercial model?** Is this a free tool, a paid app, or a service?
2. **Open source?** The model weights are on GitHub — do we release inference code too?
3. **Community maps?** Can users share AI-generated maps? Legal implications?
4. **Real-time?** Should the streaming version generate while the song plays?
5. **Difficulty beyond 5?** SynthRiders supports custom difficulties — should we?
6. **Editor timeline:** Is v2 editor Mac-only (Electron) or cross-platform (Tauri)?
7. **Spotify integration:** Do we need Spotify Developer API approval?
8. **Telemetry privacy:** Quest 3 data is personally identifiable — GDPR compliance?

---

## 14. Glossary

| Term | Definition |
|------|------------|
| **.synth** | SynthRiders beatmap file format (JSON with note list) |
| **NPS** | Notes Per Second. Density metric. Easy ≈ 2, Master ≈ 15. |
| **NMS** | Non-Maximum Suppression. Collapses clustered note predictions. |
| **MLE** | Maximum Likelihood Estimation. Standard training objective. |
| **RL** | Reinforcement Learning. Fine-tuning with reward signal. |
| **REINFORCE** | Policy gradient algorithm for discrete decisions. |
| **AUROC** | Area Under ROC Curve. Discrimination metric (1.0 = perfect). |
| **BCE** | Binary Cross-Entropy. Loss for binary classification. |
| **KL** | Kullback-Leibler divergence. Regularization against reference model. |
| **Turbo Bypass** | Skipping the Reward Engine during density descent for speed. |
| **Oracle mask** | Fixed set of musically salient frames from reference model. |
| **Scorer** | Playability classifier (Bi-LSTM + MLP). |
| **Phase 12b** | Current MLE-trained Transformer generator checkpoint. |
| **Fixed-K** | Constant number of target notes, not scaling with output. |

---

## 15. Appendix: Current State Snapshot

**Repo:** https://github.com/marc0ne-s/synth-beatmap-gen
**Active training run:** Run 3.15 (Turbo Bypass)
**Current challenge:** Model saturates to 100 NPS (every frame = note). Using supervised BCE to crash density to 2.1 NPS for Easy.
**Next milestone:** Phase 12b.Final Audit (5-difficulty pass rate validation)
**Blocker to prototype:** None. Architecture is sound. Just needs convergence and audit.
**Estimated working prototype timeline:** 1-2 weeks if Run 3.15 succeeds.

---

*Draft generated by HerBB on 2026-04-28*
*Pending Marcus review — this document is a conversation starter, not a mandate.*
